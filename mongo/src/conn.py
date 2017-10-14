#!/usr/bin/env python3
from pymongo import MongoClient
from pprint import pprint

client = MongoClient("mongodb://34.224.70.221:8080")
db=client.admin
serverStatusResult=db.command("serverStatus")
pprint(serverStatusResult)
