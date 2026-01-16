class Node:
    def __init__(self, data=None):
        self.data = data  # Node data
        self.left = None  # Left child
        self.right = None  # Right child


class BinaryTree:
    def __init__(self):
        self.root = None  # Initialize tree root

    def insert(self, data):
        if self.root is None:
            self.root = Node(data)  # Set root if tree is empty
        else:
            self._insert(data, self.root)  # Call helper to insert

    def _insert(self, data, cur_node):
        if data < cur_node.data:  # Data is smaller, go left
            if cur_node.left is None:
                cur_node.left = Node(data)  # Insert as left child
            else:
                self._insert(data, cur_node.left)  # Recurse left
        elif data > cur_node.data:  # Data is larger, go right
            if cur_node.right is None:
                cur_node.right = Node(data)  # Insert as right child
            else:
                self._insert(data, cur_node.right)  # Recurse right
        else:
            print("Value already present in tree")  # Avoid duplicates

    def display(self, cur_node):
        lines, _, _, _ = self._display(cur_node)  # Get tree structure
        for line in lines:
            print(line)  # Print tree structure

    def _display(self, cur_node):
        if cur_node.right is None and cur_node.left is None:  # Leaf node
            line = '%s' % cur_node.data
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle
        if cur_node.right is None:  # Only left child exists
            lines, n, p, x = self._display(cur_node.left)
            s = '%s' % cur_node.data
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2
        if cur_node.left is None:  # Only right child exists
            lines, n, p, x = self._display(cur_node.right)
            s = '%s' % cur_node.data
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2
        left, n, p, x = self._display(cur_node.left)  # Process left
        right, m, q, y = self._display(cur_node.right)  # Process right
        s = '%s' % cur_node.data
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:  # Adjust heights if needed
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def find_i(self, target):
        cur_node = self.root  # Start from root
        while cur_node:  # Traverse tree
            if cur_node.data == target:
                return True  # Found target
            elif target < cur_node.data:
                cur_node = cur_node.left  # Go left
            else:
                cur_node = cur_node.right  # Go right
        return False  # Not found

    def find_r(self, target):
        return self._find_r(self.root, target)  # Call recursive helper

    def _find_r(self, node, target):
        if node is None:
            return False  # Not found
        if node.data == target:
            return True  # Found target
        if target < node.data:
            return self._find_r(node.left, target)  # Recurse left
        else:
            return self._find_r(node.right, target)  # Recurse right


# Example usage
bst = BinaryTree()
for value in [4, 2, 6, 1, 3, 5, 7]:
    bst.insert(value)  # Insert values

bst.display(bst.root)  # Display tree structure

# Test search methods
print(bst.find_i(5))  # Output: True
print(bst.find_i(8))  # Output: False
print(bst.find_r(7))  # Output: True
print(bst.find_r(0))  # Output: False
