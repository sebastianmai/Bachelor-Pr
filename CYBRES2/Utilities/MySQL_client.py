#!/usr/bin/env python3
"""
This class implements the client for our database
HOW TO USE:
1. adjust config file to experiment!

2. initialize:  client_name = MySQL_client()
3. write:       client_name.write(data) where
                    data = has to be an iterable e.g np.array size len(sensors) - 1  (experiment number gets
                            specified separately); all integers
4. read:        client.read(table_name, experiment_number, timestep_start, timestep_end)
                    table_name = name of the node/table you want read from
                    experiment_number = number of the experiment
                    timestep_start = starting timestamp
                    timestep_end = ending timestep
"""

import yaml
import mysql.connector
from mysql.connector import Error



class MySQL_client:

    def __init__(self):
        with open("config.yaml", "r") as ymlfile:
            cfg = yaml.full_load(ymlfile)
        self.host_name = str(cfg["mysql"]["host"])
        self.user_name = str(cfg["mysql"]["user"])
        self.user_pw = str(cfg["mysql"]["passwd"])
        self.db_name = str(cfg["mysql"]["db"])

        self.table_name = str(cfg["experiment"]["node"])
        self.current_table = cfg["experiment"]["sensors"].split(", ")
        self.experiment_number = cfg["experiment"]["experiment_number"]

        self.connection = None
        self.__create_connection()
        self.__create_table()

    # DO NOT CALL -> should work now
    def __create_connection(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host_name,
                user=self.user_name,
                passwd=self.user_pw,
                database=self.db_name
            )
            print("Connection to MySQL DB successful")
        except Error as e:
            print(f"The error '{e}' occurred")

    def __create_table(self):
        cursor = self.connection.cursor()
        create_database_query = "CREATE TABLE IF NOT EXISTS " + str(self.table_name) + " ("
        for i in range(len(self.current_table)):
            create_database_query = create_database_query + " " + str(self.current_table[i]) + " INT,"
        create_database_query = create_database_query + " PRIMARY KEY (timeStamp)) ENGINE=InnoDB"

        try:
            cursor.execute(create_database_query)
            print("Database created successfully")
        except Error as e:
            print(f"The error '{e}' occured")

    def write(self, data):
        # sanity check! -> data should have same dim then number of sensors else wouldnt make any sense
        try:
            assert len(data) == len(self.current_table) - 1
        except AssertionError:
            print("data does not match number of sensors! Occured in " + str(self.table_name) + ". Nothing got written to the db!")
            return

        cursor = self.connection.cursor()
        insert_data = "INSERT INTO " + str(self.table_name) + " ("
        for i in range(len(self.current_table)):
            if i == 0:
                insert_data = insert_data + str(self.current_table[i])
            else:
                insert_data = insert_data + ", " + str(self.current_table[i])
        insert_data = insert_data + ") VALUES ("
        for i in range(len(data) + 1):
            if i == 0:
                insert_data = insert_data + str(self.experiment_number)
            else:
                insert_data = insert_data + ", " + str(data[i-1])
        insert_data = insert_data + ");"

        try:
            cursor.execute(insert_data)
            self.connection.commit()
            print("Query executed successfully")
        except Error as e:
            print(f"The error '{e}' occured")

    def read(self, table_name, experiment, fromTS, tilTS):
        cursor = self.connection.cursor()
        select_entries = "SELECT * FROM " + str(table_name) + " WHERE `experimentNumber` = " + str(experiment) + " AND ( `timeStamp` >= " + str(fromTS) + " AND `timeStamp` <= " + str(tilTS) + ")"

        try:
            cursor.execute(select_entries)
            return cursor.fetchall()
        except Error as e:
            print(f"The error '{e}' occured")