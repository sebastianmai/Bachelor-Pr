#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from datetime import datetime
import json
from pathlib import Path
import time
import requests
import yaml
import datetime


#http://161.53.68.176:8000/"
url = "https://stupefied-poitras.185-23-116-208.plesk.page/api/"
headers = {"Content-Type": "application/json", "Authorization": "Basic dGhpc19pc190aGVfcGFzc3dvcmQ="}


class HTTPClient(object):
    def __init__(self, node_handle, display_name) -> None:
        # Get the list of nodes currently on the website.
        self.node_handle = node_handle
        self.display_name = display_name
                
        if not self.node_exists(node_handle):
            self.add_node(node_handle, display_name)
        

    def node_exists(self, node_handle=None):
        """
        Checks if a given node already exists in the website/database

        Args:
            node_handle (str): handle of node to check, if None checks self

        Returns:
            True if node exists, False otherwise
        """
        if node_handle is None:
            node_handle = self.node_handle

        query = 'nodes'
        response = None
        while response is None:
            try:
                response = requests.request("GET", url + query, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to GET current list of nodes. Trying again.")
                time.sleep(0.5)
        parsed = json.loads(response.text)
        # print(parsed['data'])
        return any(node['handle'] == node_handle for node in parsed['data'])
        

    def add_node(self, node_handle, display_name): #node_handle has to be a string with letters!!!
        """
        Adds a new node to the website/database

        Args:
            node_handle (str): Internal identifier of the node. Important: The string has to contain letters
                                to avoid an error on the website. Don't only use numbers, even if they are 
                                formatted as string it will still lead to problems
            display_name (str): Name of the node that is shown on the website

        Returns:
            True if a successful response is received, False otherwise
        """
        if node_handle is None:
            node_handle = self.node_handle
        if display_name is None:
            display_name = self.display_name
        
        query = 'nodes'
        payload = {
            'handle': node_handle,
            'name': display_name
        }
        response = None
        while response is None:
            try:
                response = requests.request("POST", url + query, json=payload, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to POST the new node. Trying again.")
        
        if not response.ok:
            print(f"ERROR: Adding node. Status code {response.status_code}")
            return False
            
        return True

    
    def add_data_field(self, field_name, field_handle, unit, node_handle=None):
        """
        Does nothing? TODO

        Args:
            field_name (str): Name of the data field that is shown on the website
            field_handle (str): Internal identifier of the data field
            unit (str): Unit of the datatype measured. To display on the website
            node_handle (str): Is this really neaded? TODO

        Returns:
            True if a successful response is received, False otherwise
        """
        if node_handle is None:
            node_handle = self.node_handle
            
        query = 'data-field'
        payload = {
            "name": field_name,
            "handle": field_handle,
            "unit": unit
        }
        response = None
        while response is None:
            try:
                response = requests.request("POST", url + query, json=payload, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to POST the new data field. Trying again.")
        
        if not response.ok:
            print(f"ERROR: Adding data field. Status code {response.status_code}")
            return False
        
        return True
    

    def add_data(self, data, additional_sensors, node_handle=None):

        ## TODO : wait for Grants final API version, then adapt name of fields/additional sensors

        """
        Adds a single measurement set to the database
        Args:
            data (np.array): Collected dataline from the MU
            node_handle (str): Node that collected the dataset, if None, it assumes itself as collector

        Returns:
            True if a successful response is received, False otherwise
        """
        if node_handle is None:
            node_handle = self.node_handle

        data = [self.node_handle] + data.tolist()
        time = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")#datetime.fromtimestamp(data[3]).strftime("%Y-%m-%d %H:%M:%S")

        #if config_file is None:
        config_file = Path(__file__).parent.absolute() / "data_fields.yaml"

        with open(config_file) as stream:
            config = yaml.safe_load(stream)

        additional_sensors = []

        keys = [key for key in config if config[key] is True] + (additional_sensors if additional_sensors != False else [])
        data_filter = [i for i, x in enumerate(config.values()) if x] + ([j + len(config) for j in range(len(additional_sensors))] if additional_sensors != False else [])
        filtered_data = [data[i] for i in data_filter]

        query = 'sensordata'
        payload = {
            "node_handle": filtered_data[0],
            "date": time,
            "data": dict(zip(keys[1:], filtered_data[1:]))
        }
        """
            {
                "temp_pcb": "0",
                "mag_x": "0",
                "mag_y": "0",
                "mag_z": "0",
                "temp_external": "50",
                "light_external": "0",
                "humidity_external": "4",
                "differential_potential_ch1": "10",
                "differential_potential_ch2": "0",
                "rf_power_emission": "0",
                "transpiration": "0",
                "air_pressure": "0",
                "soil_moisture": "0",
                "soil_temperature": "0",
                "mu_mm": "0",
                "mu_id": "0",
                "sender_hostname": "rpi0",
                "ozone": "0",
            }
        }
        """
        response = None
        while response is None:
            try:
                response = requests.request("POST", url + query, json=payload, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to POST the new data field. Trying again.")
        
        if not response.ok:
            print(f"ERROR: Adding data. Status code {response.status_code}")
            return False
        
        return True


    def add_data2(self, node_handle=None):
        """
        Adds a single measurement set to the database
        Args:
            node_handle (str): Node that collected the dataset, if None, it assumes itself as collector
        Returns:
            True if a successful response is received, False otherwise
        """
        if node_handle is None:
            node_handle = self.node_handle
        
        query = 'sensordata'
        payload = {
            "node_handle": node_handle,
            "data": {
                "temp_external": 0,
                "light_external": 0,
                "humidity_external": 0,
                "differential_potential_ch1": 10,
                "differential_potential_ch2": 20,
                "rf_power_emission": 0,
                "transpiration": 0,
                "air_pressure": 0,
                "soil_moisture": 0,
                "soil_temperature": 0
            },
            "date": "2023-01-23 22:25:00"
        }
        response = None
        while response is None:
            try:
                response = requests.request("POST", url + query, json=payload, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to POST the new data field. Trying again.")
        
        if not response.ok:
            print(f"ERROR: Adding data. Status code {response.status_code}")
            return False
        
        return True


    def get_data(self, node_handle=None):
        """
        Returns all data entries from a specified node

        Args:
            node_handle ([str]): List of node_handles whose data shall be extracted

        Returns:
            dictionary with all data entries from all given node_handles
            Format:
            Dictionary with node_handles as keys
                List with all data entries per node_handle
                    Dictionary with metadata keys and one 'data' key
                        'data' contains a dictionary with the actual measurements 
        """
        if node_handle is None:
            node_handle = [self.node_handle]

        query = 'sensordata-multiple'
        payload = {
            "node_ids": node_handle
        }

        response = None
        while response is None:
            try:
                response = requests.request("POST", url + query, json=payload, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to retrieve data. Trying again.")

        if not response.ok:
            print(f"ERROR: Retrieving data. Status code {response.status_code}")
            return False

        parsed = json.loads(response.text)
        return parsed['data']

    def get_data_fields(self):
        """
        Returns all available data fields with handles

        Args:

        Returns:
            List of all data fields and their meta data
        """

        query = 'data-field'
        response = None
        while response is None:
            try:
                response = requests.request("GET", url + query, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to GET current list of data fields. Trying again.")
                time.sleep(0.5)

        parsed = json.loads(response.text)
        return parsed['data']


    def delete_data_field(self, data_handle):
        """
        Deletes data field with matching handle 

        Args:
            data_handle (str): handle of data field to delete

        Returns:
            
        """

        query = 'data-field/delete'
        payload = {
            "handle": data_handle
        }

        response = None
        while response is None:
            try:
                response = requests.request("POST", url + query, json=payload, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to delete data field. Trying again.")
                time.sleep(0.5)
        return True


    def delete_node(self, node_handle):
        """
        Deletes  node with matching handle 

        Args:
            node_handle (str): handle of node to delete

        Returns:
            
        """

        query = 'nodes/delete'
        payload = {
            "handle": node_handle
        }

        response = None
        while response is None:
            try:
                response = requests.request("POST", url + query, json=payload, headers=headers, timeout=2.0)
            except requests.exceptions.Timeout:
                print("Timeout while waiting to delete node. Trying again.")
                time.sleep(0.5)
        return True

        
def main():
    
    client = HTTPClient('test_node_2', 'Test node 2')
    #client.add_data2("ttyACM0")
    #print(client.get_data('test_node_2'))
    #print(client.node_exists("ttyACM0"))
    #print(client.get_data())
    #client.delete_data_field("air_pressure")
    #client.add_data_field("Air Pressure", "air_pressure", "test", node_handle=None)
    client.delete_node("ttyACM0")
    client.delete_node("test_node_2")
    #print(client.get_data(['test_node_2']))

if __name__ == '__main__':
    main()