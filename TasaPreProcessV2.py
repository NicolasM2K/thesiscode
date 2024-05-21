import os
import re

def check_dataset_format(file_path):
    errors = []
    with open(file_path, 'r', encoding='utf-8') as file:
        is_sentence = False
        
        for line_number, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            if line.startswith('[S]'):
                is_sentence = True
            elif is_sentence:
                # It could be a continuation of a sentence, skip checking
                continue
            else:
                # This should be a metadata line
                metadata_parts = line.split('[')
                metadata_parts = [part.strip('] ') for part in metadata_parts if part.strip()]

                if not metadata_parts:
                    errors.append(f"Line {line_number}: Empty or incorrect metadata format")
                else:
                    # Check for first variable format and DRP presence
                    first_variable_valid = re.match(r'[A-Za-z]+\d+.\d+.\d+', metadata_parts[0])
                    drp_present = any('DRP=' in part and re.match(r'DRP=\d+(\.\d+)?', part) for part in metadata_parts)
                    
                    if not first_variable_valid:
                        errors.append(f"Line {line_number}: First metadata variable does not match expected pattern - '{metadata_parts[0]}'")
                    if not drp_present:
                        errors.append(f"Line {line_number}: 'DRP=' variable with numeric value not found")

                is_sentence = False

    return errors

# Example usage:
input_path = './tasa.txt'

def errors():
    errors = check_dataset_format(input_path)
    if errors:
        for error in errors:
            print(error)
    else:
        print("No formatting errors found.")

def preprocess_dataset_to_files(input_file_path, output_dir, max_lines=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    current_sentences = []
    current_filename = ""
    lines_processed = 0
    
    with open(input_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if max_lines is not None and lines_processed >= max_lines:
                break
            
            lines_processed += 1
            stripped_line = line.strip()

            if stripped_line.startswith('[') and not stripped_line.startswith('[S]'):
                # This is a metadata line
                # Write the current sentences to a file if there are any
                if current_sentences and current_filename:
                    with open(os.path.join(output_dir, current_filename + '.txt'), 'w', encoding='utf-8') as out_file:
                        out_file.write('\n'.join(current_sentences))
                    current_sentences = []  # Reset for the next block of sentences
                
                # Extract filename from metadata
                metadata_parts = stripped_line.split('[')
                metadata_parts = [part.strip('] ') for part in metadata_parts if part.strip()]
                
                unique_identifier = metadata_parts[0].split(']')[0]  # Assuming the unique identifier is the first part
                drp_part = next((part for part in metadata_parts if 'DRP=' in part), 'DRP_MISSING')
                
                current_filename = f"{unique_identifier}_{drp_part}"
            elif stripped_line or current_sentences:  # If the line is not empty or we are in a sentence block
                # Either add a new sentence or continue the current sentence
                if stripped_line.startswith('[S]'):
                    current_sentences.append(stripped_line[3:].strip())
                elif current_sentences:  # Continue the current sentence if it's not the start of a new one
                    current_sentences[-1] += " " + stripped_line

        # Don't forget to save the last batch of sentences
        if current_sentences and current_filename:
            with open(os.path.join(output_dir, current_filename + '.txt'), 'w', encoding='utf-8') as out_file:
                out_file.write('\n'.join(current_sentences))

output_directory = 'C:/Users/Nicolas/Downloads/preprocessingTasa/TasaPostProcess'
max_lines_to_process = 1259849 # Change this number to whatever line count limit you want for testing. 1259849 is the total line count of the dataset.

def preprocessrequest(directory):
    # Check if there are files in the directory
    if any(os.listdir(directory)):
        # Ask for user confirmation to preprocess again
        user_input = input("The directory is populated. Are you sure you want to preprocess again? (yes/no): ")
        if user_input.lower() == 'no':
            print("Preprocessing cancelled by user.")
            return False
        elif user_input.lower() == 'yes':
            preprocess_dataset_to_files(input_path, output_directory, max_lines=max_lines_to_process)
            return False
    else:
        preprocess_dataset_to_files(input_path, output_directory, max_lines=max_lines_to_process)   
    return True

preprocessrequest(output_directory)

import binascii

def print_line_hex(file_path, line_numbers):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i in line_numbers:
                print(f'Line {i}: {line.strip()}')
                hex_representation = binascii.hexlify(line.encode('utf-8'))
                print(f'Hex: {hex_representation}')

# Example usage - replace with actual line numbers you want to check
line_numbers_to_check = [1258412, 1259452, 1259494, 1259503]
#print_line_hex(input_path, line_numbers_to_check)

def verify_filenames(directory):
    # This regular expression matches "name and date" followed by "_DRP=" and then a number, ending with ".txt"
    filename_pattern = re.compile(r'^[A-Za-z]+\d+\.\d+\.\d+_DRP=\d+(\.\d+)?\.txt$')
    incorrectly_formatted_filenames = []

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        if not filename_pattern.match(filename):
            incorrectly_formatted_filenames.append(filename)

    return incorrectly_formatted_filenames

# Example usage:
directory_path = 'C:/Users/Nicolas/Downloads/preprocessingTasa/TasaPostProcess'
incorrect_files = verify_filenames(directory_path)
if incorrect_files:
    print("Incorrectly formatted filenames found:")
    for filename in incorrect_files:
        print(filename)
else:
    print("All filenames are correctly formatted.")

def verify_file_contents(directory):
    incorrectly_formatted_files = []
    empty_files = []

    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                contents = file.read()
                
                # Check for empty files
                if not contents.strip():
                    empty_files.append(filename)
                    continue  # Skip further checks and continue with the next file

                # Check for any lines containing '[S]'
                if '[S]' in contents:
                    incorrectly_formatted_files.append(filename)
    
    return incorrectly_formatted_files, empty_files

# Example usage:

incorrect_files, empty_files = verify_file_contents(directory_path)

if incorrect_files:
    print("Files with incorrect content:")
    for filename in incorrect_files:
        print(filename)

if empty_files:
    print("Empty text files found:")
    for filename in empty_files:
        print(filename)

if not incorrect_files and not empty_files:
    print("All files are correctly formatted and non-empty.")

#There should be 37,651 files in the output directory.