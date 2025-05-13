import datetime
import random

def generate_fake_data(start_date, num_lines, category="Entertainment"):
    """Generates fake data with more realistic patterns."""
    data = []
    current_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    
    # Define possible values for the second and third columns
    value1_options = [0.1, 0.25, 0.55, 1]
    value2_suffixes = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000] 

    # Initialize a variable to track the previous value2 for smoother transitions
    prev_value2 = 0

    for _ in range(num_lines):
        # Select a random value from the defined options
        value1 = random.choice(value1_options)

        # Generate a more realistic value2 with a smoother progression
        base_value2 = random.randint(800, 1500) 
        value2_suffix = random.choice(value2_suffixes)
        value2 = (base_value2 * 1000) + value2_suffix 

        # Introduce some correlation between consecutive value2 values
        if prev_value2 > 0:
            value2 = max(value2, prev_value2 - 100000)  # Ensure some level of increase
            value2 = min(value2, prev_value2 + 200000)  # Limit the rate of change

        prev_value2 = value2

        data.append(f"{current_date},{category},{value1},{value2}")
        current_date += datetime.timedelta(days=1)

    return data

# Generate 100 lines of fake data starting from 2017-05-04
fake_data = generate_fake_data("2017-05-04", 100)

# Print the generated data
for line in fake_data:
    print(line)