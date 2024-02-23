class Piece:
    """ ガイスターの駒
    
    ガイスター駒の位置と色を管理
    ちょっとカプセル化してみた

    Attributes:
        __x (int): 駒のx位置 (0~5)
        __y (int): 駒のy位置 (0~5)
        __color (str): 駒の色（'r': 赤, 'b': 青）
        __goal (bool): ゴールしたか
        __dead (bool): 取られたか

    """

    def __init__(self, x: int, y: int, color: str):
        """ 初期化処理 """
        self.set_pos(x, y)
        self.set_color(color)

    def get_x(self):
        return self.__x

    def get_y(self):
        return self.__y

    def get_pos(self):
        return [self.__x, self.__y]

    def get_color(self):
        return self.__color

    def get_goal(self):
        return self.__goal

    def get_dead(self):
        return self.__dead

    # def set_x(self, x: int):
    #     self.__x = x
    #     self.check_dead()
    #     self.check_goal()

    # def set_y(self, y: int):
    #     self.__y = y
    #     self.check_dead()
    #     self.check_goal()

    def set_pos(self, x: int, y: int):
        self.__x = x
        self.__y = y
        self.check_dead()
        self.check_goal()

    def set_color(self, color: str):
        self.__color = color

    def set_goal(self, goal: bool):
        self.__goal = goal

    def set_dead(self, dead: bool):
        self.__dead = dead

    def move(self, player: int, move: str):
        if player == 0:
            if move == 'N':
                self.__y -= 1
            elif move == 'E':
                self.__x += 1
            elif move == 'W':
                self.__x -= 1
            elif move == 'S':
                self.__y += 1
        elif player == 1:
            if move == 'N':
                self.__y += 1
            elif move == 'E':
                self.__x -= 1
            elif move == 'W':
                self.__x += 1
            elif move == 'S':
                self.__y -= 1

    def check_goal(self) -> bool:
        if self.__x == 8 and self.__y == 8:
            self.__goal = True
        else:
            self.__goal = False
        return self.__goal

    def check_dead(self) -> bool:
        if self.__x == 9 and self.__y == 9:
            self.__dead = True
        else:
            self.__dead = False
        return self.__dead

    def print_all(self):
        print(f'position: (x, y) = ({self.__x}, {self.__y})')
        print(f'color: {self.__color}')
        print(f'goal: {self.__goal}')
        print(f'dead: {self.__dead}')


def main():
    a = Piece(1, 2, 'r')
    a.print_all()
    a.set_pos(3, 3)
    a.print_all()
    a.set_pos(8, 8)
    a.print_all()
    a.set_pos(9, 9)
    a.print_all()


if __name__ == '__main__':
    main()
