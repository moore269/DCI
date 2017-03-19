from blocks.extensions import SimpleExtension

class myTrackBest(SimpleExtension):

    def __init__(self, record_name, notification_name=None,
                 choose_best=min, **kwargs):
        self.record_name = record_name
        if not notification_name:
            notification_name = record_name + "_best_so_far"
        self.notification_name = notification_name
        self.best_name = "best_" + record_name
        self.choose_best = choose_best
        self.params=None
        kwargs.setdefault("after_epoch", True)
        super(myTrackBest, self).__init__(**kwargs)

    #update our current snapshot
    def update_Snapshot(self):
        self.params = self.main_loop.model.get_parameter_values()

    def set_best_model(self):
        self.main_loop.model.set_parameter_values(self.params)

    def do(self, which_callback, *args):
        if self.params==None:
            self.update_Snapshot()

        current_value = self.main_loop.log.current_row.get(self.record_name)
        if current_value is None:
            return
        best_value = self.main_loop.status.get(self.best_name, None)
        if (best_value is None or
                (current_value != best_value and
                 self.choose_best(current_value, best_value) ==
                 current_value)):
            self.main_loop.status[self.best_name] = current_value
            self.main_loop.log.current_row[self.notification_name] = True
            self.update_Snapshot()

