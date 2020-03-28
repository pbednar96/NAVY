import turtle

ACTION = 4
ITER = 3
LENGTH = 10

def lsystem(axiom, rules, iterations):
    #create final axiom

    for _ in range(iterations):
        final_axioms = ''
        for item in axiom:
            if item == "F":
                final_axioms += rules
            else:
                final_axioms += item
        axiom = final_axioms
    return axiom


def turtle_route(axiom, angel):
    # draw axiom with turtle

    window = turtle.Screen()
    window.title("L system")
    line = turtle.Turtle()
    stack = []

    line.left(90)

    for item in axiom:
        if item == "F":
            line.forward(LENGTH)
        elif item == "+":
            line.right(angel)
        elif item == "-":
            line.left(angel)
        elif item == '[':
            stack.append((line.heading(), line.pos()))
        elif item == ']':
            heading, position = stack.pop()
            line.penup()
            line.goto(position)
            line.setheading(heading)
            line.pendown()
    turtle.done()


def main():
    if ACTION == 1:
        axiom = "F+F+F+F"
        rules = "F+F-F-FF+F+F-F"
        angle = 90

    if ACTION == 2:
        axiom = "F++F++F"
        rules = "F+F--F+F"
        angle = 60

    if ACTION == 3:
        axiom = "F"
        rules = "F[+F]F[-F]F"
        angle = 180 / 7

    if ACTION == 4:
        axiom = "F"
        rules = "FF+[+F-F-F]-[-F+F+F]"
        angle = 180/8

    if ACTION == 5:
        axiom = "F"
        rules = "F[+FF][-FF]F[-F][+F]F"
        angle = 35

    final_axiom = lsystem(axiom, rules, ITER)
    print(final_axiom)
    turtle_route(final_axiom, angle)

if __name__ == "__main__":
    main()
