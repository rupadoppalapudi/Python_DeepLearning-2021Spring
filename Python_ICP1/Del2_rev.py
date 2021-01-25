# 2. Write a python program for the following:
# Input the string “Python” as a list of characters from console, delete at least 2 characters, reverse the resultant string and print it.

# users enters string input
text = list(input("Enter the text to be processed: "))
# excluding/ deleting first character of the string
result = "".join(text[1:])
# reverse the string
result_rev = result[::-1]
# excluding/ deleting first character of the string after string reversal
output = "".join(result_rev[1:])

# displaying the string output after deleting 2 characters and reversing the resultant string
print("Output string obtained: ", output)


