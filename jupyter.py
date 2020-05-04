

def Section():
    def __init__(self, mode, safe_value=0, from_list=None, to_list=None):
        self.mode = mode
        self.from_list = from_list
        self.to_list = to_list
        self.safe_value = safe_value

    def push():
        pass

    def insert(self, insert_section: Section):

        if self.to_list is None:
            self.to_list = [insert_section]
        else:
            self.to_list.append(insert_section)





