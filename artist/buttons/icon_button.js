import React from "react";
import {Icon} from "react-native-elements";


export default function IconButton({name, type, onPress}) {
    return (
        <Icon
            reverse
            name={name}
            type={type}
            size={20}
            color={'#517fa4'}
            onPress={onPress}
        />
    )
}