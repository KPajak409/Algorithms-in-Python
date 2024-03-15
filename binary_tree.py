#%%
class Node:
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val


def printInOrder(root):

    if root:
        printInOrder(root.left)

        print(root.val, end=" ")

        printInOrder(root.right)


def printPreOrder(root):
    if root:
        print(root.val, end=" ")

        printPreOrder(root.left)

        printPreOrder(root.right)


def printPostOrder(root):
    if root:
        printPostOrder(root.left)

        printPostOrder(root.right)

        print(root.val, end=" ")


def printLevelOrder(root):
    h = height(root)
    for i in range(h):
        printCurrentLevel(root, i)


def printCurrentLevel(root, level):
    if root is None:
        return
    if level == 0:
        print(root.val, end=" ")
    elif level > 0:
        printCurrentLevel(root.left, level-1)
        printCurrentLevel(root.right, level-1)


def height(node):
    if node is None:
        return 0
    else:
        lheight = height(node.left)
        rheight = height(node.right)

        if lheight > rheight:
            return lheight+1
        else:
            return rheight+1
        

if __name__ == "__main__":
    root = Node(1)
    root.left = Node(2)
    root.right = Node(3)
    root.left.left = Node(4)
    root.left.right = Node(5)
    root.right.left = Node(6)
    root.right.right = Node(7)
 
    # Function call
    print("In-order traversal of binary tree is")
    #Postorder 4,2,5,1,6,3,7
    printInOrder(root)

    print("\n\nPre-order traversal of binary tree is")
    #Postorder 1,2,4,5,3,6,7
    printPreOrder(root)

    print("\n\nPost-order traversal of binary tree is")
    #Postorder 4,5,2,6,7,3,1
    printPostOrder(root)

    print("\n\nLevel-order traversal of binary tree is")
    #Postorder 1,2,3,4,5,6,7
    printLevelOrder(root)

# %%
