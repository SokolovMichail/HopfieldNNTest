import numpy as np
import random
class RM_generator:
    def generate_new_rm(self,auditories,groups,teachers,days,lessons_per_day):
        rm = np.zeros((4,4,4))
        number_of_lessons = round(days * lessons_per_day * 0.1 * auditories * groups * teachers)
        for i in range(number_of_lessons):
            x = random.randint(0,auditories-1)
            y = random.randint(0,groups-1)
            z = random.randint(0,teachers-1)
            rm[x,y,z]+=1
        return rm

test = RM_generator()
for i in range(10):
    print(test.generate_new_rm(4,4,4,2,5))