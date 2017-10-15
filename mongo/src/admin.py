#!/usr/bin/env python3
from pymongo import MongoClient
from pprint import pprint

client = MongoClient("mongodb://34.224.70.221:8080")

# create database
db_pred = client['predictions']
print(db_pred)

# insert data
