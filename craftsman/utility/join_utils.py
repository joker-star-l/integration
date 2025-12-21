import numpy as np
import duckdb
import psycopg2
import pymonetdb
from psycopg2 import OperationalError
from pandas.api.types import is_integer_dtype, is_bool_dtype, is_string_dtype, is_float_dtype

from craftsman.base.defs import DBDataType

def insert_db(db, table_name, cols, data):
    if db == 'duckdb':
        duckdb_file_path = '/root/volume/duckdb/mydb'
        conn = duckdb.connect(database=duckdb_file_path, read_only=False)
        # create table
        cols_sql = ','.join([f'{col} {cols[col]}' for col in cols])
        create_sql = f'DROP TABLE IF EXISTS {table_name}; CREATE TABLE {table_name} ({cols_sql});'
        conn.execute(create_sql)
        # temp_tables.append(table_name)
        # insert data
        slot_sql = ','.join(['?'] * len(data[0]))
        insert_sql = f'INSERT INTO {table_name} VALUES ({slot_sql});'
        conn.executemany(insert_sql, data)
        conn.commit()
        conn.close()
    
    elif db == 'postgresql':
        # database parameters
        dbname = 'postgres'
        user = 'postgres'
        host = 'localhost'  
        port = 5432  

        # connect to pg
        try:
            connection = psycopg2.connect(
                dbname=dbname,
                user=user,
                host=host,
                port=port
            )
            cursor = connection.cursor()
        except OperationalError as e:
            print(f"The error '{e}' occurred")

        # create table
        cols_sql = ','.join([f'{col} {cols[col]}' for col in cols])
        create_sql = f'DROP TABLE IF EXISTS {table_name}; CREATE TABLE {table_name} ({cols_sql});'
        try:
            cursor.execute(create_sql)
            connection.commit()
        except OperationalError as e:
            print(f"The error '{e}' occurred")

        # insert data
        slot_sql = ','.join(['%s'] * len(data[0]))
        insert_sql = f'INSERT INTO {table_name} VALUES ({slot_sql});'
        cursor.executemany(insert_sql, data)
        connection.commit()

        # close connect
        cursor.close()
        connection.close()
        
    elif db == 'monetdb':
        conn = pymonetdb.connect(database='craftsman', username='monetdb', password='monetdb')
        cur = conn.cursor()
        # create table
        cols_sql = ','.join([f'{col} {cols[col]}' for col in cols])
        create_sql = f'DROP TABLE IF EXISTS {table_name}; CREATE TABLE {table_name} ({cols_sql});'
        cur.execute(create_sql)
        conn.commit()
        # temp_tables.append(table_name)
        # insert data
        slot_sql = ','.join(['%s'] * len(data[0]))
        insert_sql = f'INSERT INTO {table_name} VALUES ({slot_sql});'
        cur.executemany(insert_sql, data)
        conn.commit()
        conn.close()



def df_type2db_type(df_type, db):
    if db in ('duckdb', 'postgresql'):
        if is_string_dtype(df_type):
            return DBDataType.VARCHAR.value
        elif df_type == np.int8:
            return DBDataType.SMALLINT.value
        elif is_integer_dtype(df_type):
            return DBDataType.INT.value
        elif is_float_dtype(df_type):
            return DBDataType.FLOAT.value
        elif is_bool_dtype(df_type):
            return DBDataType.BOOLEAN.value
    
    elif db == 'monetdb':
        if is_string_dtype(df_type):
            return DBDataType.VARCHAR512.value
        elif df_type == np.int8:
            return DBDataType.SMALLINT.value
        elif is_integer_dtype(df_type):
            return DBDataType.INT.value
        elif is_float_dtype(df_type):
            return DBDataType.FLOAT.value
        elif is_bool_dtype(df_type):
            return DBDataType.BOOLEAN.value
    
    return None


