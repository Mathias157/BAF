
$ifi not %FICTDEM% == yes $goto NoFictDem
* Include fictive demand on heat
$onmulti
$if     EXIST '../data/ANTBALM_FICTDHUSER.inc' $INCLUDE         '../data/ANTBALM_FICTDHUSER.inc';
$if not EXIST '../data/ANTBALM_FICTDHUSER.inc' $INCLUDE '../../base/data/ANTBALM_FICTDHUSER.inc';
$offmulti
$label NoFictDem
