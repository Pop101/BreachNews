INSERT INTO Corporations (
  corpName,
  orgType,
  numCustomers,
  numEmployees
)
VALUES (?, ?, ?, ?);

INSERT INTO DataBreaches (
  corpName,
  breachDate,
  numRecords,
  method,
  numArticles,
  sensitivity
)
VALUES (?, ?, ?, ?, ?, ?);

INSERT INTO NewsArticles (
  pubDate,
  title,
  lang,
  country,
  publisher,
  category,
  breachId
)
VALUES (?, ?, ?, ?, ?, ?, ?);