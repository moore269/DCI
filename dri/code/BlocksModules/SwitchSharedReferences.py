from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.algorithms import DifferentiableCostMinimizer
from blocks.monitoring.evaluators import AggregationBuffer
from blocks.extensions.monitoring import MonitoringExtension
from blocks.monitoring.aggregation import (_DataIndependent, Mean,
                                           TakeLast, MonitoredQuantity)
from blocks.utils import dict_subset
from blocks.graph import ComputationGraph
import logging
import theano

logger = logging.getLogger(__name__)
#switches references to shared input batches on the fly

class SwitchSharedReferences(SimpleExtension):

    def __init__(self, sharedBatch, sharedData, **kwargs):
        kwargs.setdefault("after_batch", True)
        super(SwitchSharedReferences, self).__init__(**kwargs)
        #need weird names or else weird pickle error
        self.sharedB = {}
        self.sharedD = {}
        #need this or else we get weird pickle error
        for key in sharedBatch:
            self.sharedB[key]=sharedBatch[key]
        for key in sharedData:
            self.sharedD[key]=sharedData[key]


    def do(self, callback_name, *args):
        batch, from_user = self.parse_args(callback_name, args)
        i = batch[0]['int_stream_To']
        #print(i)
        for key in self.sharedB:
            if key!='nodeID':
                self.sharedB[key].set_value(self.sharedD[key][i].get_value(borrow=True), borrow=True)
            else:
                self.sharedB[key] = self.sharedD[key][i]


class MonitoredQuantityBuffer(object):

    def __init__(self, quantities):
        self.quantities = quantities
        requires = []
        for quantity in quantities:
            requires += quantity.requires
        self.requires = list(set(requires))
        self._initialized = False

        self.quantity_names = [q.name for q in self.quantities]
        self._computation_graph = ComputationGraph(self.requires)
        self.inputs = self._computation_graph.inputs

    def initialize(self):
        """Initialize the quantities."""
        self._initialized = True
        for quantity in self.quantities:
            quantity.initialize()

    def get_aggregated_values(self):
        """Readout the accumulated values."""
        if not self._initialized:
            raise Exception("To readout you must first initialize, then"
                            "process batches!")
        else:
            ret_vals = [q.readout() for q in self.quantities]
            return dict(zip(self.quantity_names, ret_vals))

    def accumulate_quantities(self, numerical_values):
        """Accumulate the results for every batch."""
        if not self._initialized:
            raise Exception("To readout you must first initialize, then"
                            "process batches!")
        else:
            for quantity in self.quantities:
                quantity.accumulate(
                    *[numerical_values[self.requires.index(requirement)]
                        for requirement in quantity.requires])

class myDatasetEvaluator(object):
    def __init__(self, variables, updates=None, sharedBatch=None, sharedData=None):
        #need weird names or else weird pickle error
        self.sharedB = {}
        self.sharedD = {}
        #need this or else we get weird pickle error
        for key in sharedBatch:
            self.sharedB[key]=sharedBatch[key]
        for key in sharedData:
            self.sharedD[key]=sharedData[key]

        theano_variables = []
        monitored_quantities = []
        for variable in variables:
            if isinstance(variable, MonitoredQuantity):
                monitored_quantities.append(variable)
            else:
                theano_variables.append(variable)
        self.theano_variables = theano_variables
        self.monitored_quantities = monitored_quantities
        variable_names = [v.name for v in variables]
        if len(set(variable_names)) < len(variables):
            raise ValueError("variables should have different names")
        self.theano_buffer = AggregationBuffer(theano_variables)
        self.monitored_quantities_buffer = MonitoredQuantityBuffer(
            monitored_quantities)
        self.updates = updates
        self._compile()

    def _compile(self):
        inputs = []
        outputs = []
        updates = None
        if self.theano_buffer.accumulation_updates:
            updates = OrderedDict()
            updates.update(self.theano_buffer.accumulation_updates)
            inputs += self.theano_buffer.inputs
        if self.updates:
            # Handle the case in which we dont have any theano variables
            # to evaluate but we do have MonitoredQuantity
            # that may require an update of their own
            if updates is None:
                updates = self.updates
            else:
                updates.update(self.updates)
        inputs += self.monitored_quantities_buffer.inputs
        outputs = self.monitored_quantities_buffer.requires


        self.unique_inputs = list(set(inputs))
        self._accumulate_fun = theano.function(self.unique_inputs,
                                               outputs,
                                               updates=updates)


    def initialize_aggregators(self):
        self.theano_buffer.initialize_aggregators()
        self.monitored_quantities_buffer.initialize()

    def process_batch(self, batch):
        try:
            input_names = [v.name for v in self.unique_inputs]
            batch = dict_subset(batch, input_names)
        except KeyError:
            reraise_as(
                "Not all data sources required for monitoring were"
                " provided. The list of required data sources:"
                " {}.".format(input_names))
        if self._accumulate_fun is not None:
            numerical_values = self._accumulate_fun(**batch)
            self.monitored_quantities_buffer.accumulate_quantities(
                numerical_values)

    def get_aggregated_values(self):
        values = self.theano_buffer.get_aggregated_values()
        values.update(
            self.monitored_quantities_buffer.get_aggregated_values())
        return values

    def evaluate(self, data_stream):
        self.initialize_aggregators()
        if self._accumulate_fun is not None:
            for batch in data_stream.get_epoch_iterator(as_dict=True):
                i=batch['int_stream_From']
                #print(i)
                for key in self.sharedB:
                    if key!='nodeID':
                        self.sharedB[key].set_value(self.sharedD[key][i].get_value(borrow=True), borrow=True)
                    else:
                        self.sharedB[key] = self.sharedD[key][i]
                self.process_batch(batch)
            for key in self.sharedB:
                if key!='nodeID':
                    self.sharedB[key].set_value(self.sharedD[key][0].get_value(borrow=True), borrow=True)
                else:
                    self.sharedB[key] = self.sharedD[key][0]
        
        else:
            logger.debug(
                'Only data independent variables were given,'
                'will not iterate the over data!')

        return self.get_aggregated_values()


class DataStreamMonitoringShared(SimpleExtension, MonitoringExtension):

    PREFIX_SEPARATOR = '_'

    def __init__(self, variables, data_stream, updates=None, sharedBatch=None, sharedData=None, **kwargs):
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(DataStreamMonitoringShared, self).__init__(**kwargs)
        #need weird names or else weird pickle error
        self.sharedB = {}
        self.sharedD = {}
        #need this or else we get weird pickle error
        for key in sharedBatch:
            self.sharedB[key]=sharedBatch[key]
        for key in sharedData:
            self.sharedD[key]=sharedData[key]
        self._evaluator = myDatasetEvaluator(variables, updates, self.sharedB, self.sharedD)
        self.data_stream = data_stream

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")

