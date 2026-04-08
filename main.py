


assignment_grade = int(input("What is the assignment grade? "))
quiz_grade = int(input("Enter the quiz grade? "))
test_grade = int(input("Enter the test grade? "))

overall_grade = 0.2*assignment_grade + 0.3*quiz_grade + 0.5*test_grade
print(f"Your overall grade is {overall_grade}.")
