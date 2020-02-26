(function () {
    'use strict';
    angular.module('app')
        .constant('endpoints', {
            COMPANY: {
                FETCH_ALL: 'hierarchy/angularData',
                FETCH_ALL_FLAT: 'hierarchy/angularStructureList'
            },
            REPORTING: {
                FETCH_ALL: 'hierarchy/angularReportingData',
                FETCH_ALL_FLAT: 'hierarchy/angularReportingList'
            }
        });
})();