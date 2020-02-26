(function () {
    'use strict';
    angular.module('app').factory('dataService', ['$http','endpoints', dataService]);

    function dataService($http,endpoints) {
        return {
            browseCompanyHierarchy: browseCompanyHierarchy,
            browseReportingHierarchy: browseReportingHierarchy,
            browseCompanyHierarchyList: browseCompanyHierarchyList,
            browseReportingHierarchyList: browseReportingHierarchyList
        };

        function browseReportingHierarchy() {
            return $http.get(http_path + endpoints.REPORTING.FETCH_ALL).then(function (response) {
                console.log(response);
                return response.data.data;
            }).catch(function (error) {
                console.error(error);
            });
        }
        function browseCompanyHierarchy() {
            return $http.get(http_path + endpoints.COMPANY.FETCH_ALL).then(function (response) {
                console.log(response);
                return response.data.data;
            }).catch(function (error) {
                console.error(error);
            });
        }
        
        function browseCompanyHierarchyList() {
            return $http.get(http_path + endpoints.COMPANY.FETCH_ALL_FLAT).then(function (response) {
                console.log(response);
                return response.data.data;
            }).catch(function (error) {
                console.error(error);
            });
        }

        function browseReportingHierarchyList() {
            return $http.get(http_path + endpoints.REPORTING.FETCH_ALL_FLAT).then(function (response) {
                console.log(response);
                return response.data.data;
            }).catch(function (error) {
                console.error(error);
            });
        }
    }
})();