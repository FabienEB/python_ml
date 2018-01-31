# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:12:05 2018

@author: fbaker
"""

import sqlite3
import pandas as pd


def create_parameter_scan_table(table_name, columns, conn):
    """ creates the table for setups to test
    """
    c= conn.cursor()
    query = " CREATE TABLE "
    query += table_name
    query += " ("
    query += ",".join(columns)
    query += ")"

    c.execute(query)
    c.close()


def add_values_to_table(table_name, values, conn):
    """ adds new values to tables
    """

    c= conn.cursor()
    query = "INSERT INTO "
    query += table_name
    query += " VALUES ("
    query += ",".join( "?" * len(values[0]))
    query += ")"
    # print (query)
    # print(values, len(values))
    c.executemany(query, values)
    conn.commit()
    # print( pd.read_sql_query("select * from " + table_name, conn))

    c.close()


def fetch_table_data_into_df(table_name, conn):

    return pd.read_sql_query("select * from " + table_name, conn)
        

    
