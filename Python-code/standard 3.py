class Node:
    def __init__(self, dataval=None):  # Constructor to initialize a node
        self.dataval = dataval  # Store data in the node
        self.nextval = None  # Initialize the next pointer as None

class SLinkedList:
    def __init__(self):  # Constructor to initialize the linked list
        self.headval = None  # Head of the linked list starts as None

    def listprint(self):  # Method to print all nodes in the list
        printval = self.headval  # Start with the head of the list
        while printval is not None:  # Iterate through the list until the end
            print(printval.dataval)  # Print the data of the current node
            printval = printval.nextval  # Move to the next node

    def AtBeginning(self, newdata):  # Method to add a node at the beginning
        NewNode = Node(newdata)  # Create a new node
        NewNode.nextval = self.headval  # Link the new node to the current head
        self.headval = NewNode  # Update the head to the new node

    def AtEnd(self, newdata):  # Method to add a node at the end
        NewNode = Node(newdata)  # Create a new node
        if self.headval is None:  # If the list is empty
            self.headval = NewNode  # Set the new node as the head
            return  # Exit the method
        last = self.headval  # Start with the head of the list
        while last.nextval:  # Traverse to the last node
            last = last.nextval  # Move to the next node
        last.nextval = NewNode  # Link the last node to the new node

    def Insert(self, val_before, newdata):  # Method to insert a node after a given value
        if val_before is None:  # If no value to insert after is provided
            print("No node to insert after")  # Print error message
            return  # Exit the method
        else:
            current = self.headval  # Start with the head of the list
            while current is not None:  # Traverse the list
                if current.dataval == val_before:  # If the current node matches the target value
                    NewNode = Node(newdata)  # Create a new node
                    NewNode.nextval = current.nextval  # Link the new node to the next node
                    current.nextval = NewNode  # Link the current node to the new node
                    return  # Exit the method after inserting
                current = current.nextval  # Move to the next node
            print(f"Value '{val_before}' not found in the list.")  # Print error if value not found

# Initialize the linked list
list = SLinkedList()  # Create a new singly linked list
list.headval = Node("Mon")  # Create and set the head node with "Mon"

e2 = Node("Tue")  # Create a node with "Tue"
e3 = Node("Thur")  # Create a node with "Thur"
e4 = Node("Fri")  # Create a node with "Fri"
e5 = Node("Sat")  # Create a node with "Sat"
list.headval.nextval = e2  # Link "Mon" to "Tue"
e2.nextval = e3  # Link "Tue" to "Thur"
e3.nextval = e4  # Link "Thur" to "Fri"
e4.nextval = e5  # Link "Fri" to "Sat"

list.Insert("Tue", "Weds")  # Insert "Weds" after "Tue"
list.AtEnd("Sun")  # Add "Sun" at the end of the list

list.listprint()  # Print all nodes in the list
