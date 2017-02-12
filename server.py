import zerorpc,gevent,signal,sys
import utils as utils
import json

class KerasRPC(object):
    def train(self, message):
        print(message)
        m = json.loads(message)
        intervals = m["keystrokes"]
        user = m["user"]
        newUser = m["newUser"]

        if newUser:
            utils.singleTrain(intervals,user,newUser)
            returnMessage = "CREATED"
            Y = 1
        else:
            classes, prob = utils.singleTest(intervals,user)
            utils.save_stat(user,intervals,classes[0],prob,newUser)
            if classes[0] == 1:
                # utils.singleTrain(intervals,user,newUser)
                returnMessage = "VALID"
                Y = 1
            else:
                returnMessage = "INVALID"
                Y = 0

        utils.saveIntervals(intervals, Y, user, newUser)
        back = {'stat':utils.get_stat(user),
               'message':returnMessage }
        return back



s = zerorpc.Server(KerasRPC())
s.bind("tcp://0.0.0.0:4242")
gevent.signal(signal.SIGTERM, s.stop)
gevent.spawn(s.run)
# s.run()
while True:
  gevent.sleep(1)

# print("zpc stopped"); sys.stdout.flush()