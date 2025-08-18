* Balmorel and Antares coupling addon
$ifi not %FICTDEM% == yes $goto NoFictDem
$onmulti
$if     EXIST '../data/ANTBALM_FICTDH.inc' $INCLUDE         '../data/ANTBALM_FICTDH.inc';
$if not EXIST '../data/ANTBALM_FICTDH.inc' $INCLUDE '../../base/data/ANTBALM_FICTDH.inc';
$offmulti
$label NoFictDem