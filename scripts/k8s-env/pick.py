#!/usr/bin/env python3

import sys,os
import curses

class Item(object):

    items = {
            0: "   fake,~/.kube/config-fake",
            1: "   online,~/.kube/config-online",
        }
    def __init__(self, choose=0):
        self.__choose = choose
        self.choose(choose)
        self.chosed = False

    def get_current(self):
        return self.items.get(self.__choose).split(",")[1]

    def get_info(self):
        return self.items.get(self.__choose).split(",")[0][4:]

    def clear(self):
        s = self.items.get(self.__choose)
        s = "    " + s[4:]
        self.items[self.__choose] = s

    def choose(self, c):
        if not isinstance(c, int) or c > len(self.items):
            return
        s = self.items.get(c)
        s = " => " + s[4:]
        self.items[c] =s
        self.__choose = c

    def down(self):
        self.clear()
        self.choose((self.__choose + 1) % len(self.items))


    def up(self):
        self.clear()
        # print((self.__choose - 1) % len(self.items))
        self.choose((self.__choose - 1) % len(self.items))

it = Item()

def draw_menu(stdscr):
    k = 0
    cursor_x = 0
    cursor_y = 0

    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()

    # Start colors in curses
    curses.start_color()
    curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE)

    # Loop where k is the last character pressed
    while (k != ord('q')):

        # Initialization
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        if k == curses.KEY_DOWN:
            it.down()
            cursor_y = cursor_y + 1
        elif k == curses.KEY_UP:
            it.up()
            cursor_y = cursor_y - 1
        # elif k == curses.KEY_RIGHT:
        #     cursor_x = cursor_x + 1
        # elif k == curses.KEY_LEFT:
        #     cursor_x = cursor_x - 1
        elif k == 10:
            it.chosed = True
            return

        cursor_x = max(0, cursor_x)
        cursor_x = min(width-1, cursor_x)

        cursor_y = max(0, cursor_y)
        cursor_y = min(height-1, cursor_y)

        # Declaration of strings
        title = "Choose Cluster"[:width-1]
        subtitle = "Written by Clay McLeod"[:width-1]
        keystr = "Last key pressed: {}".format(k)[:width-1]
        statusbarstr = "Press 'q' to exit | STATUS BAR | Pos: {}, {}".format(cursor_x, cursor_y)
        if k == 0:
            keystr = "No key press detected..."[:width-1]

        # Centering calculations
        start_x_title = int((width // 2) - (len(title) // 2) - len(title) % 2)
        start_x_subtitle = int((width // 2) - (len(subtitle) // 2) - len(subtitle) % 2)
        # start_x_keystr = int((width // 2) - (len(keystr) // 2) - len(keystr) % 2)
        start_y = int((height // 2) - 2)

        # Rendering some text
        whstr = "Width: {}, Height: {}".format(width, height)
        stdscr.addstr(0, 0, whstr, curses.color_pair(1))

        # Render status bar
        stdscr.attron(curses.color_pair(3))
        stdscr.addstr(height-1, 0, statusbarstr)
        stdscr.addstr(height-1, len(statusbarstr), " " * (width - len(statusbarstr) - 1))
        stdscr.attroff(curses.color_pair(3))

        # Turning on attributes for title
        stdscr.attron(curses.color_pair(2))
        stdscr.attron(curses.A_BOLD)

        # Rendering title
        stdscr.addstr(start_y, start_x_title-1, title)

        # Turning off attributes for title
        stdscr.attroff(curses.color_pair(2))
        stdscr.attroff(curses.A_BOLD)

        # Print rest of text
        for k,v in it.items.items():
            stdscr.addstr(start_y + k +2,start_x_subtitle,v.split(",")[0])

        # stdscr.addstr(start_y + 1, start_x_subtitle, subtitle)
        # stdscr.addstr(start_y + 3, (width // 2) - 2, '-' * 4)
        # stdscr.addstr(start_y + 5, start_x_keystr, keystr)
        stdscr.move(cursor_y, cursor_x)

        # Refresh the screen
        stdscr.refresh()

        # Wait for next input
        k = stdscr.getch()

def main():
    curses.wrapper(draw_menu)
    tmp_file = sys.argv[1]
    if not it.chosed:
        return
    with open(tmp_file,'w+') as f:
        f.writelines(f'export PS1="({it.get_info()}) ${{PS1:-}}"\n')
        f.writelines(f'export KUBECONFIG={it.get_current()}\n')
        f.writelines(f"rm -f {tmp_file}")
if __name__ == "__main__":
    main()
