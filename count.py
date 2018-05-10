"""Counts the average number of words in training datasets"""
from string import punctuation

files = ["data/agreed.polarity", "data/disagreed.polarity"]
all_examples = []

for f in files:
    with open(f, 'r') as file:
        for line in file.readlines():
            cleaned_line = ""
            for char in line:
                if char not in punctuation:
                    cleaned_line += char
            all_examples.append(cleaned_line)

overall_words_count = 0
number_of_examples = len(all_examples)

for line in all_examples:
    line_length = len(line.split(" "))
    overall_words_count += line_length

count = overall_words_count/number_of_examples
print(f"Average words count: {round(count)}")
