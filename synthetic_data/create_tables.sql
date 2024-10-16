 -- Dates are TEXT "YYYY-MM-DD HH:MM:SS.SSS"

CREATE TABLE IF NOT EXISTS Corporations(
  corpName TEXT PRIMARY KEY,
  orgType TEXT,
  numCustomers INTEGER DEFAULT 0,
  numEmployees INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS DataBreaches(
  breachId INTEGER PRIMARY KEY,
  corpName TEXT,
  breachDate TEXT,
  numRecords INTEGER DEFAULT 0,
  method TEXT,
  numArticles INTEGER DEFAULT 0,
  sensitivity REAL DEFAULT 0,
  FOREIGN KEY(corpName) REFERENCES Corporations(corpName)
);

CREATE TABLE IF NOT EXISTS NewsArticles(
  articleId INTEGER PRIMARY KEY,
  pubDate TEXT,
  title TEXT,
  lang TEXT,
  country TEXT,
  publisher TEXT,
  category TEXT,
  breachId INTEGER,
  FOREIGN KEY(breachId) REFERENCES DataBreaches(breachId)
);