# --------------
# Code starts here

class_1 = ['Geoffrey Hinton','Andrew Ng','Sebastian Raschka','Yoshua Bengio']

class_2 = ['Hilary Mason','Carla Gentry','Corinna Cortes']

new_class = class_1 + class_2
print("New Class:",new_class)

new_class1 = ['Peter Warden']
new_class= new_class + new_class1
print(new_class)

new_class.remove('Carla Gentry')
print(new_class)

# Code ends here


# --------------
# Code starts here
courses = {'Math':65,'English':70,'History':80,'French':70,'Science':60}
total = courses['Math'] + courses['English'] + courses['History'] + courses['French'] + courses['Science']
print(total)

percentage = 100.0 * total / 500
print(percentage)



# Code ends here


# --------------
# Code starts here

mathematics = {'Geoffrey Hinton':78,'Andrew Ng':95,'Sebastian Raschka':65,'Yoshua Benjio':50,'Hilary Mason':70,'Corinna Cortes':66,'Peter Warden':75}

topper = max(mathematics,key = mathematics.get)
print (topper)

# Code ends here  


# --------------
# Given string
topper = 'andrew ng'


# Code starts here

first_name = topper.split()[0]
print(first_name)
last_name = topper.split()[1]
print(last_name)
full_name = last_name + " " + first_name
print(full_name)
certificate_name = full_name.upper()
print(certificate_name)

# Code ends here

