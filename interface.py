import PySimpleGUI as psg

init_complete = False

class Dictionaries:
    def __init__(self,teacher_list,auditory_list,learning_groups_list):
        self.teacher_number_dict = {}
        self.number_teacher_dict = {}
        self.auditory_number_dict = {}
        self.number_auditory_dict = {}
        self.learninggroups_number_dict = {}
        self.number_learninggroup_dict = {}
        t = 0
        for i in teacher_list:
            self.teacher_number_dict[i] = t
            self.number_teacher_dict[t] = i
        t = 0
        for i in auditory_list:
            self.auditory_number_dict[i] = t
            self.number_auditory_dict[t] = i
        t = 0
        for i in learning_groups_list:
            self.learninggroups_number_dict[i] = t
            self.number_learninggroup_dict[t] = i


def load_files():
    teacher_list = []
    auditories_list = []
    learninggroupslist = []
    with open('work_files/teachers.txt') as f:
        teacher_list = f.read().splitlines()
    with open('work_files/auditories.txt') as f:
        auditories_list = f.read().splitlines()
    with open('work_files/groups.txt') as f:
        learninggroupslist = f.read().splitlines()
    return ((teacher_list,auditories_list,learninggroupslist),Dictionaries(teacher_list,auditories_list,learninggroupslist))

interface_lists, dictionaries = load_files()



layout = [
    [psg.Radio("Самописная нейронная сеть Хопфилда","GROUP_ALGORITHM")],
    [psg.Radio("Генетический алгоритм", "GROUP_ALGORITHM")],
    [psg.Radio("Нейронная сеть Хопфилда на базе TensorFlow","GROUP_ALGORITHM")],
    [psg.Text('Преподаватели'),psg.Text('Аудитории'),psg.Text('Учебные группы')],
    [psg.Listbox(interface_lists[0]),psg.Listbox(interface_lists[1]),psg.Listbox(interface_lists[2])],
    [psg.Submit("Запустить")],
    [psg.Output(size=(88, 20))]

]

window = psg.Window('Curricula_Generator', layout)

while True:                             # The Event Loop
    event, values = window.read()
    # print(event, values) #debug
    if event in (None, 'Exit', 'Cancel'):
        break