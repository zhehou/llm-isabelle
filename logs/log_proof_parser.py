import sys

# Check if at least one command-line argument is provided.
# sys.argv[0] is the script name, so we need a length of at least 2.
if len(sys.argv) < 2:
    print("Usage: python clean_string.py \"<your_string_to_process>\"")
    print("Error: No input string was provided.")
    sys.exit(1) # Exit the script indicating an error

# The first argument provided by the user is at index 1
input_string = sys.argv[1]

# 1. Replace the escaped quote '\"' with a simple quote '"'
# 2. Replace the escaped newline '\n' with an actual newline character
processed_string = input_string.replace('\\"', '"').replace('\\n', '\n')

# Print the final, cleaned-up string
print(processed_string)