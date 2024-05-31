/*
Q: What is a transaction?

A transaction is a single unit of work.

SQL Server operates in the following transaction modes:

Autocommit transactions:-
Each individual statement is a transaction.

Explicit transactions:-
Each transaction is explicitly started with the BEGIN TRANSACTION 
statement and explicitly ended with a COMMIT or ROLLBACK statement.

Note: Auto-Commit is turned off in HME SSMS

*/

//Commit transaction
const query = `BEGIN TRANSACTION;
DELETE FROM HumanResources.JobCandidate
    WHERE JobCandidateID = 13;
COMMIT TRANSACTION;`;
