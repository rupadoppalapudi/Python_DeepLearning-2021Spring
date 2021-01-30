# python program to find the wordcount in a file for each line and then print the output.
# Finally store the output back to the file.

# open the input file
file_in = open("input_wcount",'r')

# defining dictionary to store key-value pairs of word and count
word_count = {}

# looping through all the lines in the file by reading and splitting the lines into words
for txt in file_in.read().split():
    if txt not in word_count:
        word_count[txt] = 1
    else:
        word_count[txt] += 1

# closing the file after reading all the lines
file_in.close()

# declaring variable to store the output
output_txt = ''

# storing the words and count in output file
for word, count in word_count.items():
    output_txt += word + ': ' + str(count) + '\n'

# open the output file, write data into the file and close the file
output_file = open("output_wcount", "w")
output_file.write(output_txt)
output_file.close()

print("\n Word count is completed successfully. Output is stored in the output_wcount file!!")