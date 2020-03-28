from math import pi

import turtle

ACTION = 3


def lsystem(axiom, rules, iterations):
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
    window = turtle.Screen()
    window.title("L system")
    line = turtle.Turtle()
    for item in axiom:
        if item == "F":
            line.forward(10)
        elif item == "+":
            line.right(angel)
        elif item == "-":
            line.left(angel)

    turtle.done()


def main():
    if ACTION == 1:
        axiom = "F+F+F+F"
        rules = "F+F-F-FF+F+F-F"
        angle = 90
        final_axiom = lsystem(axiom, rules, 2)
        print(final_axiom)
        turtle_route(final_axiom, angle)

    if ACTION == 2:
        axiom = "F++F++F"
        rules = "F+F--F+F"
        angle = 60
        final_axiom = lsystem(axiom, rules, 3)
        print(final_axiom)
        turtle_route(final_axiom, angle)


if __name__ == "__main__":
    main()
