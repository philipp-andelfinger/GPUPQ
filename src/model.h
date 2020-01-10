#include "config.h"

//
// Model Interface:
//

// Initialize the model before any events are processed.
// This can be used to fill the queue with initial events.
void model_init();

// Execute the next chunk of parallel events.
// This gets called by the simulator forever, as long as none of the force exit conditions defined in config.h are met.
void model_handle_next();

// Returns the sum of all events processed by the LPs so far.
long model_get_events();

// Clean up the model before exiting the simulator.
void model_finish();
