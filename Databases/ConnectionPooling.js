/*
Q1: What is Connection Pooling?

In software engineering, a connection pool is a cache of database connections maintained 
so that the connections can be reused when future requests to the database are required.

An important concept to understand when using this library (mssql-node) is Connection Pooling as this 
library uses connection pooling extensively. As one Node JS process is able to handle multiple requests at once, 
we can take advantage of this long running process to create a pool of database connections for reuse; 
this saves overhead of connecting to the database for each request (as would be the case in something like PHP, 
where one process handles one request).

*/

//Connect to global connection pool using mssql
const sql = require('mssql')
const config = { }

// run a query against the global connection pool
function runQuery(query) {
  // sql.connect() will return the existing global pool if it exists or create a new one if it doesn't
  return sql.connect(config).then((pool) => {
    return pool.query(query)
  })
}


//instantiate a connection pool
const appPool = new sql.ConnectionPool(config)

/*
Q2: How to create multiple connection pools?

sql.connect(config) connects to the global conn pool if it exists else creates a new conn pool with the config obj
Otherwise if connect is called with some other config, the new config gets ignored.

With sql.ConnectionPool(config) you can create objects of with different configs, each of which is maintained as a 
separate pool of each. 
*/

//connection.js
const pool1 = new sql.ConnectionPool(config);
const poolConnect1 = pool1.connect();

const pool2 = new sql.ConnectionPool(config2);
const poolConnect2 = pool2.connect();

const connectToConfig = async (pool, poolConnect, query) => {
    await poolConnect;
    const result = await pool.request().input().query(query) || pool.request().query(query);
    return result;
}
module.exports = {
    pool1,
    poolConnect1,
    pool2,
    poolConnect2,
    connectToConfig
}

//connectionUser.js
connectToConfig(pool1, poolConnect1, query1)// query for database 1 
connectToConfig(pool2, poolConnect2, query2)// query for database 2