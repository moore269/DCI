from blocks.extensions import FinishAfter
from sys import maxint

class FinishIfNoImprovementEpsilonAfter(FinishAfter):
    """Iterations not supported!!"""

    """Stop after improvements have ceased for a given period.
    Parameters
    ----------
    notification_name : str
        The name of the log record to look for which indicates a new
        best performer has been found.  Note that the value of this
        record is not inspected.
    iterations : int, optional
        The number of iterations to wait for a new best. Exactly one of
        `iterations` or `epochs` must be not `None` (default).
    epochs : int, optional
        The number of epochs to wait for a new best. Exactly one of
        `iterations` or `epochs` must be not `None` (default).
    patience_log_record : str, optional
        The name under which to record the number of iterations we
        are currently willing to wait for a new best performer.
        Defaults to `notification_name + '_patience_epochs'` or
        `notification_name + '_patience_iterations'`, depending
        which measure is being used.
    Notes
    -----
    By default, runs after each epoch. This can be manipulated via
    keyword arguments (see :class:`blocks.extensions.SimpleExtension`).
    """
    def __init__(self, notification_name, epochs=None, epsilon=None,
                 patience_log_record=None, **kwargs):
        if (epochs is None) or (epsilon is None):
            raise ValueError("Need exactly epochs and epsilon need "
                             "to be specified")
        self.notification_name = notification_name
        self.epochs = epochs
        self.epsilon = epsilon
        self.prevBest = maxint
        kwargs.setdefault('after_epoch', True)
        self.last_best_epoch = None
        if patience_log_record is None:
            self.patience_log_record = (notification_name + '_patience' +
                                        ('_epochs' if self.epochs is not None
                                         else '_iterations'))
        else:
            self.patience_log_record = patience_log_record
        super(FinishIfNoImprovementEpsilonAfter, self).__init__(**kwargs)

    def update_best(self):
        if self.last_best_epoch==None:
            self.last_best_epoch=0

        #we need to do better by at least epsilon to count
        if self.prevBest -self.main_loop.log.current_row['valid_cost'] > self.epsilon:
            self.last_best_epoch = self.main_loop.log.status['epochs_done']

        #need to update previous best
        self.prevBest=self.main_loop.status['best_valid_cost'] 

    def do(self, which_callback, *args):
        self.update_best()

        if self.epochs is not None:
            since = (self.main_loop.log.status['epochs_done'] -
                     self.last_best_epoch)
            patience = self.epochs - since     

        self.main_loop.log.current_row[self.patience_log_record] = patience
        if patience == 0:
            super(FinishIfNoImprovementEpsilonAfter, self).do(which_callback,
                                                       *args)