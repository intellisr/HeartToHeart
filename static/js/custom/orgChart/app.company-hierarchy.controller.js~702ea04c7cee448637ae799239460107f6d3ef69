(function () {
    'use strict';
    angular.module('app').controller('CompanyHierarchyController', ['$scope', 'dataService', '$rootScope', function ($scope, dataService, $rootScope) {
        var vm = this;
        vm.actionString = "[{\"name\":\"Add\",\"icon\":\"hie-add-mark-icon\",\"callBack\":\"add\"},{\"name\":\"Manage\",\"icon\":\"hie-group-mark-icon\",\"callBack\":\"abn\"},{\"name\":\"Edit\",\"icon\":\"hie-edit-mark-icon\",\"callBack\":\"edit\"},{\"name\":\"Delete\",\"icon\":\"hie-delete-mark-icon\",\"callBack\":\"del\"}]";
        vm.treeData = {};
        vm.treeIsReady = false;
        var lastEditedNode = {};
        var lastClickedNode = {};
        vm.options = {
            word: 'hello',
            from: '0',
            until: '2',
            toolTip: 'true',
            toolTipDirection: 'top'
        };
        vm.abn = abn;
        vm.del = del;
        vm.edit = edit;
        vm.add = add;
        onInit();

        function onInit() {
            dataService.browseCompanyHierarchy().then(function (resp) {
                vm.treeData = resp;
                vm.treeIsReady = true;
            }).catch(function (err) {
                console.error(err);
            });
            dataService.browseCompanyHierarchyList().then(function (resp) {
                vm.searchableList = resp;
                // vm.treeIsReady = true;
            }).catch(function (err) {
                console.error(err);
            });

            registerJQueryTriggerListeners();
        }

        function registerJQueryTriggerListeners() {
            $(document).on('app::node::updated', {}, function (event, name) {
                console.log('nodeUpdated', name);
                //TODO add callback when nodeUpdated
                lastEditedNode.name = name;
                $rootScope.$broadcast('app:nodes:update::after', {lastEditedNode: lastEditedNode});
            });

            $(document).on('app::node::inserted', {}, function (event, node) {

                $rootScope.$broadcast('app:nodes:add::after', {parentNode: lastClickedNode, lastAddedNode: node});
            });
        }

        function abn(node) {
            hierarchyDetails(node.id);
        }

        function del(node) {
            deleteLayer(node.id);
        }

        function add(node) {
            lastClickedNode = node;
            addLayer(node.id, 'insert', node.depth);
        }

        function edit(node) {
            lastEditedNode = node;
            addLayer(node.id, 'update', node.depth);
        }

        function addLayer(parent_id, status, depth) {
            var $modal = $('#myModal');
            $modal.load(http_path + '/hierarchy/AddCompanyLayerBox', {
                    parent_id: parent_id,
                    status: status,
                    depth: depth
                },
                function () {
                    $modal.modal('show');
                    form.reset();
                });
        }


        function hierarchyDetails(parent_id) {
            var $modal = $('#myModal');
            $modal.load(http_path + '/hierarchy/CompanyHierarchyDetails', {parent_id: parent_id},
                function () {
                    $modal.modal('show');
                    form.reset();
                });
        }

        function deleteLayer(id) {
            swal({
                    title: "Are you sure?",
                    text: "You will not be able to recover this imaginary Data!",
                    type: "warning",
                    showCancelButton: true,
                    confirmButtonColor: "#DD6B55",
                    confirmButtonText: "Yes, delete it!",
                    cancelButtonText: "No, cancel plx!",
                    closeOnConfirm: true,
                    closeOnCancel: true
                },
                function (isConfirm) {
                    if (isConfirm) {
                        $.ajax({
                            url: http_path + 'index.php/Hierarchy/DeleteCompanyLayer',
                            data: "id=" + id,
                            type: 'POST',
                            success: function (res) {
                                var obj = jQuery.parseJSON(res);
                                if (obj.code == 200) {
                                    $rootScope.$broadcast('app:nodes:delete', {nodeID: id});
                                } else {
                                    swal(obj.msg, "");
                                }
                            }
                        });
                        // swal("Deleted!", "Your imaginary file has been deleted.", "success");

                    } else {
                        swal("Cancelled", "This Roll Type Assing Users :)", "error");
                    }
                });
        }
    }]);
})();