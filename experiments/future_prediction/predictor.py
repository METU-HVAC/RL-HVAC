import pandas as pd
import time
class FutureObservationProvider:
    def __init__(self, csv_path, n):
        self.n = n
        self.data = pd.read_csv(csv_path)

        # Build fast lookup mapping: time -> row
        self.time_to_row = {
            row['time']: row for _, row in self.data.iterrows()
        }

        # Build ordered list of times to preserve order
        self.time_list = list(self.data['time'])

        # Map time to index for fast search
        self.time_to_index = {time: idx for idx, time in enumerate(self.time_list)}

        # Store last row for padding
        self.last_row = self.time_to_row[self.time_list[-1]]

    def get_future_observations(self, current_time):
        if current_time not in self.time_to_index:
            raise ValueError(f"Time '{current_time}' not found in dataset.")

        current_idx = self.time_to_index[current_time]

        future_observations = []
        for i in range(1, self.n + 1):
            target_idx = current_idx + i

            if target_idx < len(self.time_list):
                target_time = self.time_list[target_idx]
                row = self.time_to_row[target_time]
            else:
                # Out of bounds: use last available row
                row = self.last_row

            future_observations.append({
                'time': row['time'],
                'outdoor_temperature': row['outdoor_temperature'],
                'people_occupant': row['people_occupant'],
            })

        return future_observations
    
if __name__ == "__main__":
    csv_path = './logs/whole_year.csv'
    #Log the load time and predict time
    predictor = FutureObservationProvider(csv_path, n=4)

    current_time = '12-31 23:30'
    
    future = predictor.get_future_observations(current_time)

    for i, obs in enumerate(future, 1):
        print(f"t+{i}:", obs)