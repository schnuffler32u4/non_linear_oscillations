import sys

def replace_equations(file_name, dracula=False):
    with open(file_name, "r") as file:
        content = file.read()
    
    # Split the content by "$$"
    equations = content.split("$$")
    
    # Replace odd occurrences of "$$" with "\begin{equation}" and even occurrences with "\end{equation}"
    for i in range(len(equations)):
        if i % 2 == 1:
            equations[i] = "\n\\begin{equation}" + equations[i] + "\\end{equation}\n"
        else:
            # Replace markdown headers with unnumbered LaTeX sections
            lines = equations[i].strip().split("\n")
            for j in range(len(lines)):
                if lines[j].startswith("#"):
                    level = lines[j].count("#")
                    title = lines[j][level:].strip()
                    if level == 1:
                        lines[j] = "\\section*{" + title + "}"
                    elif level == 2:
                        lines[j] = "\\subsection*{" + title + "}"
                    elif level == 3:
                        lines[j] = "\\subsubsection*{" + title + "}"
                    else:
                        lines[j] = "\\paragraph*{" + title + "}"
            equations[i] = "\n".join(lines)
    
    # Join the content back
    new_content = "".join(equations)
    
    # Write the new content to a new file
    with open("intro.tex", "r") as file:
        preamble = file.read()
    
    new_content = preamble + new_content + "\n\end{document}"

    with open("output" + ".tex", "w") as file:
        
        file.write(new_content)

argument = sys.argv

if len(argument)==1:
    print("Please provide a filename")
else:
    file = argument[1]

    if len(argument)==3:
        if argument[2] == 'dracula':
            dracula=True

    else: 
        dracula = False

replace_equations(file)
