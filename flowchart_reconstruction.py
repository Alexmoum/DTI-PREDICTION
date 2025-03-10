from graphviz import Digraph

# Create a directed graph (flowchart)
flowchart = Digraph('Flowchart', format='png')
flowchart.attr(rankdir='TB')  # Top-to-bottom layout (use 'LR' for left-to-right)

# Start (Oval)
flowchart.attr('node', shape='oval')
flowchart.node('Start', 'Start')

# Initialize reconstructed_sequence as an empty list (Rectangle)
flowchart.attr('node', shape='rectangle')
flowchart.node('Initialize', 'Initialize reconstructed_sequence as an empty list')

# Loop through protein_sequence with index i (Rectangle)
flowchart.node('Loop', 'Loop through protein_sequence with index i')

# Retrieve triplet from reverse_word_dict using index i (Rectangle)
flowchart.node('Retrieve', 'Retrieve triplet from reverse_word_dict using index i')

# Decision Diamond: Check conditions
flowchart.attr('node', shape='diamond')
flowchart.node('Decision', 'Check conditions')

# Group conditions into a single decision node
with flowchart.subgraph() as cond:
    cond.attr(rank='same')  # Keep conditions aligned
    cond.node('Skip1', 'Skip (continue loop)', shape='rectangle')
    cond.node('Skip2', 'Skip (continue loop)', shape='rectangle')
    cond.node('AddFull', 'Add full triplet ', shape='rectangle')
    cond.node('AddLast', 'Add last amino acid ', shape='rectangle')

# Connect decision node to conditions
flowchart.edge('Decision', 'Skip1', label='i == 0')
flowchart.edge('Decision', 'Skip2', label='i == len(protein_sequence) - 1')
flowchart.edge('Decision', 'AddFull', label='i == 1')
flowchart.edge('Decision', 'AddLast', label='Otherwise')

# Skip edges (back to loop start)
flowchart.edge('Skip1', 'Loop')
flowchart.edge('Skip2', 'Loop')

# Add edges for AddFull and AddLast
flowchart.edge('AddFull', 'EndLoop')
flowchart.edge('AddLast', 'EndLoop')

# End loop (Rectangle)
flowchart.attr('node', shape='rectangle')
flowchart.node('EndLoop', 'End loop')

# End (Oval)
flowchart.attr('node', shape='oval')
flowchart.node('End', 'End')

# Connect nodes
flowchart.edge('Start', 'Initialize')
flowchart.edge('Initialize', 'Loop')
flowchart.edge('Loop', 'Retrieve')
flowchart.edge('Retrieve', 'Decision')
flowchart.edge('EndLoop', 'End')

# Save the flowchart as an image file
flowchart.render('flowchart_compact', format='png', cleanup=True)

print("Compact flowchart saved as 'flowchart_compact.png'")