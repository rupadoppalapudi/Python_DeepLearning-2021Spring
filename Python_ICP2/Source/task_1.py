# conversion of heights from feet to cm for a list of n students

# defining 2 lists to store heights(feet and cm) of students
height_feet = []
height_cm = []

# read the number of students
n = int(input("Please enter the number of students: "))

i = 1
# storing the heights of each student in feet into a list
while i <= n:
    height_feet.append(float(input(f"Enter height of student_{i} : ")))
    i = i + 1
print("\nList of students height in feet", height_feet)

# converting height in feet to cm and saving the values in new list
for x in height_feet:
    height_cm.append(round(x * 30.48, 1))

# displaying the output list to the user
print("List of students height in cm", height_cm)


