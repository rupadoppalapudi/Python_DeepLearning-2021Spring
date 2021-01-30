# reducing the given non-negative integer to zero based on odd and even value

# read the user input - number
num = int(input("Please enter the number: "))

# number of steps
count = 0

# reducing the given number to zero
while num > 0:
    # reduce the number to half if even
    if num % 2 == 0:
        num = num / 2
    # subtract 1 from the number if odd
    else:
        num = num - 1
    count = count + 1

# displaying the output to the user
print("\nNumber of steps required: ", count)