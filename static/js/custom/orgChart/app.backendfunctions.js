function AddHierarchyLayer(status) {
    console.log(status);
    showInfoMessage('logoUp');
    sendData('companyhierarchy-form', '', 'Hierarchy/AddCompanyHierarchyLayer', function (res) {
        var obj;
        if(jQuery.isPlainObject( res )){
            obj = res;
        }else if(typeof res === 'string'){
            obj = jQuery.parseJSON(res);
        }

        if (obj.code == 200) {
            var name = extractFormData();
            showSuccessMessage(obj.msg, 'logoUp');
            if(status==='update'){
                $( document ).trigger( "app::node::updated", [name] );
            }else if(status==='insert'){
                // $( document ).trigger( "app::node::updated", [name] );
            }

        } else {
            showErrorMessage(obj.msg, 'logoUp');
        }
    });
}

