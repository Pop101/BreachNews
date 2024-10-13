import os
import warnings

def embedding_pipeline(model_name='nomic-ai/nomic-embed-text-v1.5', tokenizer_name='bert-base-uncased', use_gpu=True, **kwargs):
    from transformers import AutoTokenizer, AutoModel, Pipeline
    import torch
    import torch.nn.functional as F

    class EmbeddingPipeline(Pipeline):
        def __init__(self, model_name='nomic-ai/nomic-embed-text-v1.5', tokenizer_name='bert-base-uncased', **kwargs):
            # Load component models
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, safe_serialization=True, rotary_scaling_factor=2)
            
            # Load to GPU if available
            if use_gpu:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
            model.eval()
            
            # Initialize pipeline
            super().__init__(
                model     = model,
                tokenizer = tokenizer,
                device    = 0 if torch.cuda.is_available() and use_gpu else -1,
                **kwargs
            )

        def preprocess(self, inputs):
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
                
            inputs = [
                "search_document: " + i if not i.startswith("search_document: ") and not i.startswith("search_query: ")
                else i
                for i in inputs
            ]
            
            return self.tokenizer(inputs, padding=True, truncation=True, return_tensors='pt')

        def _forward(self, model_inputs):
            with torch.no_grad():
                model_output = self.model(**model_inputs)
            embeddings = EmbeddingPipeline.mean_pooling(model_output, model_inputs['attention_mask'])
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
        
        def _sanitize_parameters(self, **kwargs):
            return {}, {}, {}

        def postprocess(self, model_outputs):
            return model_outputs
        
        # These methods are for langchain compatibility:
        
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            output = list()
            for text in texts:
                inputs = self.preprocess(text) # Note: if error: self.preprocess("search_document: " + str(text))
                model_output = self._forward(inputs)
                model_output = self.postprocess(model_output)
                output.append(model_output.numpy().reshape(-1).tolist())
            return output
        
        def embed_query(self, query: str) -> list[float]:
            inputs = self.preprocess(query)
            model_output = self._forward(inputs) # Note: if error: self._forward(inputs)
            model_output = self.postprocess(model_output)
            return model_output.numpy().reshape(-1).tolist()
        
        @staticmethod
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
    return EmbeddingPipeline(model_name=model_name, tokenizer_name=tokenizer_name, **kwargs)

def load_pipeline(model_name, quantize = False, **kwargs):
    """
    Loads a pipeline from /projects/bdata/llm_models or a custom path.
    By default, the pipeline is set up for text generation with 
    temperature = 0.0,
    repetition penalty = 1.1,
    max tokens = 1024
    """
    
    import transformers
    import torch
    
    model_path = os.path.abspath(os.path.join('/projects/bdata/llm_models', model_name)) if not os.path.sep in model_name else model_name
    if not os.path.exists(model_path):
        raise ValueError(f"Model {model_name} not found in ({model_path})")
    
    kwargs = {
        **{
            'task': 'text-generation',
            'repetition_penalty': 1.1,
            'max_new_tokens': 1024,
        },
        **kwargs
    }
    
    if 'temperature' in kwargs and kwargs['temperature'] <= 0.0:
        warnings.warn('Warning: temperature <= 0.0 is not supported for text generation, setting do_sample=False and removing temperature.')
        del kwargs['temperature']
        kwargs['do_sample'] = False
    
    bnb_config = None
    if torch.cuda.is_available():
        if quantize:
            bnb_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                #bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
    else:
        warnings.warn('Warning: CUDA is not available, using CPU')
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()
    
    return transformers.pipeline(
        model=model,
        tokenizer=transformers.AutoTokenizer.from_pretrained(model_path),
        #return_full_text=True,  # langchain expects the full text
        **kwargs
    )