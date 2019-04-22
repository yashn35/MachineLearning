def convertFloat(string):
    try:
        return float(string)
    except ValueError:
        print("Sorry, this is not a number")
print(convertFloat("y"))