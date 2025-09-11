
* Include market value as estimated by Antares
$ifi not %MARKETVAL%==yes $goto NoMarketValue
$if     EXIST '../data/ANTBALM_MARKETVAL.inc' $INCLUDE         '../data/ANTBALM_MARKETVAL.inc';
$if not EXIST '../data/ANTBALM_MARKETVAL.inc' $INCLUDE '../../base/data/ANTBALM_MARKETVAL.inc';
$label NoMarketValue

$ifi not %CAPCRED%==yes $goto NoCapCred
* Include Electricity Generation Capacity Credits
$if     EXIST '../data/ANTBALM_GCAPCRED.inc' $INCLUDE         '../data/ANTBALM_GCAPCRED.inc';
$if not EXIST '../data/ANTBALM_GCAPCRED.inc' $INCLUDE '../../base/data/ANTBALM_GCAPCRED.inc';

* Include Electricity Transmission Capacity Credits
$if     EXIST '../data/ANTBALM_XCAPCRED.inc' $INCLUDE         '../data/ANTBALM_XCAPCRED.inc';
$if not EXIST '../data/ANTBALM_XCAPCRED.inc' $INCLUDE '../../base/data/ANTBALM_XCAPCRED.inc';

* Include Electricity Reserve Margin
$if     EXIST '../data/ANTBALM_RESMAR.inc' $INCLUDE         '../data/ANTBALM_RESMAR.inc';
$if not EXIST '../data/ANTBALM_RESMAR.inc' $INCLUDE '../../base/data/ANTBALM_RESMAR.inc';

* Include Maximum Electricity Demand
$if     EXIST '../data/ANTBALM_MAXDEM.inc' $INCLUDE         '../data/ANTBALM_MAXDEM.inc';
$if not EXIST '../data/ANTBALM_MAXDEM.inc' $INCLUDE '../../base/data/ANTBALM_MAXDEM.inc';

$ifi not %H2CAPCRED%==yes $goto NoH2CapCred
* Include Hydrogen Transmission Capacity Credits
$if     EXIST '../data/ANTBALM_XH2CAPCRED.inc' $INCLUDE         '../data/ANTBALM_XH2CAPCRED.inc';
$if not EXIST '../data/ANTBALM_XH2CAPCRED.inc' $INCLUDE '../../base/data/ANTBALM_XH2CAPCRED.inc';

* Include Hydrogen Reserve Margin
$if     EXIST '../data/ANTBALM_H2RESMAR.inc' $INCLUDE         '../data/ANTBALM_H2RESMAR.inc';
$if not EXIST '../data/ANTBALM_H2RESMAR.inc' $INCLUDE '../../base/data/ANTBALM_H2RESMAR.inc';

* Include Maximum Hydrogen Demand
$if     EXIST '../data/ANTBALM_H2MAXDEM.inc' $INCLUDE         '../data/ANTBALM_H2MAXDEM.inc';
$if not EXIST '../data/ANTBALM_H2MAXDEM.inc' $INCLUDE '../../base/data/ANTBALM_H2MAXDEM.inc';
$label NoH2CapCred
$label NoCapCred

* Include H2
$ifi not %FICTDEM%==yes $goto NoFictDem
$if     EXIST '../data/ANTBALM_FICTDH2.inc' $INCLUDE         '../data/ANTBALM_FICTDH2.inc';
$if not EXIST '../data/ANTBALM_FICTDH2.inc' $INCLUDE '../../base/data/ANTBALM_FICTDH2.inc';
$label NoFictDem
 