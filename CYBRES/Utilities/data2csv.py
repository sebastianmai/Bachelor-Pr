#!/usr/bin/env python3
import csv
import yaml
from datetime import datetime
from pathlib import Path


class data2csv:

    def __init__(self, file_path, file_name, additionalSensors, config_file=None):

        Path(file_path).mkdir(parents=True, exist_ok=True) # make new directory
        self.file_path = file_path
        self.file_name = file_name
        self.additionalSensors = additionalSensors

        self.csvfile = open(file_path + file_name, 'w')
        self.csvwriter = csv.writer(self.csvfile)

        if additionalSensors == "energy":
            header = ['timestamp', "bus_voltage_solar", "current_solar", "bus_voltage_battery", "current_battery", "ip0", "ip1", "ip2", "ip3"]
            self.csvwriter.writerow(header)
            self.csvfile.close()
        else:

            if config_file is None:
                config_file = Path(__file__).parent.absolute() / "data_fields.yaml"

            with open(config_file) as stream:
                config = yaml.safe_load(stream)

            # Data fields are loaded in their original order by default
            # and we always want to add our timestamp.
            header = ['timestamp'] + [key for key in config if config[key] is True] + (additionalSensors if additionalSensors != False else [])
            self.csvwriter.writerow(header)
            self.csvfile.close()

            self.filter = [i for i, x in enumerate(config.values()) if x] + ([j + len(config) for j in range(len(additionalSensors))] if additionalSensors != False else [])

    def close_file(self):
        self.csvfile.close()

    def write2csv(self, data):
        try:
            if self.additionalSensors == "energy":
                timestamp = datetime.fromtimestamp(data[0])
                filtered_data = data[1:]
            else:
                timestamp = datetime.fromtimestamp(data[3]).strftime("%Y-%m-%d %H:%M:%S")
                filtered_data = [data[i] for i in self.filter]

            data4csv = [timestamp] + filtered_data
            self.csvfile = open(self.file_path + self.file_name, 'a')
            self.csvwriter = csv.writer(self.csvfile)
            self.csvwriter.writerow(data4csv)
            self.csvfile.close()
            
        except Exception as e:
            return e