# Object Recognition fMRI task

## Creating the design files
To create the design files for a new subject, run the following command:

``python make_designs.py -subject wlsubjXXX``

New files will appear in design/wlsubjXXX

## Executing the task
To run the first session of the experiment, run the following command:

``python OR.py -subject wlsubjXXX -session 1``

Each run of the first session for that subject will execute in order. See the script for optional command line arguments.
