class Node:
    def __init__(self, data=None):
        self.data = data  # The node's data value
        self.left = None  # Left child pointer
        self.right = None  # Right child pointer

class BinaryTree:
    def __init__(self):
        self.root = None  # Root of the tree, initially empty

    def insert(self, data):
        if self.root is None:
            self.root = Node(data)  # Insert root if the tree is empty
        else:
            self._insert(data, self.root)  # Call the recursive insert helper method

    def _insert(self, data, cur_node):
        if data < cur_node.data:
            if cur_node.left is None:
                cur_node.left = Node(data)  # Insert in left subtree
            else:
                self._insert(data, cur_node.left)  # Recur to the left
        elif data > cur_node.data:
            if cur_node.right is None:
                cur_node.right = Node(data)  # Insert in right subtree
            else:
                self._insert(data, cur_node.right)  # Recur to the right
        else:
            print("Value already present in tree")  # Prevent duplicates

    def display(self, cur_node):
        lines, _, _, _ = self._display(cur_node)  # Get tree lines for display
        for line in lines:
            print(line)  # Print each line of the tree

    def _display(self, cur_node):
        if cur_node.right is None and cur_node.left is None:
            line = '%s' % cur_node.data  # Leaf node, display its value
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        if cur_node.right is None:
            lines, n, p, x = self._display(cur_node.left)
            s = '%s' % cur_node.data
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s  # Only left child
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        if cur_node.left is None:
            lines, n, p, x = self._display(cur_node.right)
            s = '%s' % cur_node.data
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '  # Only right child
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        left, n, p, x = self._display(cur_node.left)
        right, m, q, y = self._display(cur_node.right)
        s = '%s' % cur_node.data
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)  # Adjust left side to match height
        elif q < p:
            right += [m * ' '] * (p - q)  # Adjust right side to match height
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def remove(self, target):
        if self.root is None:  # Case 0: Empty tree
            print("Tree is empty.")
            return False

        parent = None
        node = self.root

        # Traverse to find the target node
        while node and node.data != target:
            parent = node
            if target < node.data:
                node = node.left
            else:
                node = node.right

        if node is None:  # Case 1: Target not found
            print(f"Node {target} not found.")
            return False

        print(f"Removing node {target}...")

        # Case 2: Node has no children (leaf node)
        if node.left is None and node.right is None:
            if parent is None:  # Target is root node
                self.root = None
            elif parent.left == node:
                parent.left = None
            else:
                parent.right = None
            return True

        # Case 3: Node has left child only
        if node.left is not None and node.right is None:
            if parent is None:  # If the node is the root
                self.root = node.left
            elif parent.left == node:
                parent.left = node.left
            else:
                parent.right = node.left
            return True

        # Case 4: Node has right child only
        if node.left is None and node.right is not None:
            if parent is None:  # If the node is the root
                self.root = node.right
            elif parent.left == node:
                parent.left = node.right
            else:
                parent.right = node.right
            return True

        # Case 5: Node has two children
        successor_parent = node
        successor = node.right

        # Find the in-order successor (smallest node in the right subtree)
        while successor.left:
            successor_parent = successor
            successor = successor.left

        # Replace node's data with successor's data
        node.data = successor.data

        # Remove successor node
        if successor_parent.left == successor:
            successor_parent.left = successor.right  # Successor has no left child
        else:
            successor_parent.right = successor.right  # Successor has no left child

        return True


# Example usage
bst = BinaryTree()
# Insert nodes into the tree
bst.insert(4)
bst.insert(2)
bst.insert(6)
bst.insert(1)
bst.insert(3)
bst.insert(5)
bst.insert(7)

# Display the tree before removal
print("Tree before removal:")
bst.display(bst.root)

# Remove a node (for example, remove the node with value 3)
bst.remove(3)

# Display the tree after removal
print("\nTree after removal of 3:")
bst.display(bst.root)

# Remove another node (for example, remove the node with value 6)
bst.remove(6)

# Display the tree after removal
print("\nTree after removal of 6:")
bst.display(bst.root)

# Try removing a non-existent node
bst.remove(10)
