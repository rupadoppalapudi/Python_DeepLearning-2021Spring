# creating Employee class
class Employee:
    count_emp = 0                             # count of Employees
    total_emp_salary = 0                      # total salary of employees

    # Creating a constructor to initialize name, family, salary, department
    def __init__(self, name, family, salary, department):
        self.name = name
        self.family = family
        self.salary = salary
        self.department = department
        Employee.count_emp += 1               # increment Employee count by 1 for each employee
        Employee.total_emp_salary += salary   # calculate employee salary

    # creating a function to calculate average salary
    def avg_salary(self):
        average_salary = Employee.total_emp_salary / Employee.count_emp
        return average_salary

# creating a Fulltime Employee class that inherits the properties of Employee class
class FullTime_Emp(Employee):
    # Creating a constructor to initialize name, family, salary, department, age
    def __init__(self, name, family, salary, department, age):
        Employee.__init__(self, name, family, salary, department)
        self.age = age

# creating the instances of Employee class
emp_1 = Employee("Rupa", "Doppalapudi", 80000, "ABAP")
emp_2 = Employee("Sri", "DK", 40000, "BI")

# accessing members(variables and functions) of class
print("\nEmployee count: ", Employee.count_emp)
print("Total Employees salary: ", Employee.total_emp_salary)
print("The average salary of employees: ", emp_2.avg_salary())

# creating the instances of Fulltime Employee class
ft_Emp_1 = FullTime_Emp("Kalyan", "Kilaru", 90000, "Basis", 25)

# accessing members(variables and functions) of class
print("\nEmployee count: ", ft_Emp_1.count_emp)
print("Total Employees salary: ", ft_Emp_1.total_emp_salary)
print("The average salary of employees: ", ft_Emp_1.avg_salary())




