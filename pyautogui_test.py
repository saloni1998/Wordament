import pyautogui as p
#p.PAUSE=1
def func():
	p.moveTo(1507,532)
	p.dragTo(1507,532,button='left')
	#p.click(1097,395)
	p.dragTo(1370,395,button='left')
	#p.moveTo(1234,532)
	#p.click(1234,532)
	p.dragTo(1234,532,button='left')
	#p.click(1370,395)
#func()


positions={}

x_axis = [1097,1233,1370,1507]
y_axis = [395,532,671,805]
 
def func_arpan():
    p.moveTo(1507,532)
    p.mouseDown(button='left')
    #p.mouseDown()
    p.moveTo(1370,395)
    p.mouseDown()
    p.moveTo(1234,532)
    p.mouseDown()
    p.mouseUp()


def do_click(word):
    for i in range(len(word)):
        print(word[i][1])
        p.moveTo(x_axis[int(word[i][0])],y_axis[int(word[i][1])])
        p.mouseDown(button='left')
    p.mouseUp()
    
#func_arpan()
do_click(['10','11','12'])
