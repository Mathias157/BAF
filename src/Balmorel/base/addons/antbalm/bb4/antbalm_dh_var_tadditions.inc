
* Balmorel and Antares coupling addon
$ifi not %FICTDEM% == yes $goto NoFictDem
$onmulti
$if     EXIST '../data/ANTBALM_FICTDH_VAR_T.inc' $INCLUDE         '../data/ANTBALM_FICTDH_VAR_T.inc';
$if not EXIST '../data/ANTBALM_FICTDH_VAR_T.inc' $INCLUDE '../../base/data/ANTBALM_FICTDH_VAR_T.inc';
$offmulti
$label NoFictDem