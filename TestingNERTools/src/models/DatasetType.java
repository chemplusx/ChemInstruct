package models;

import java.util.HashMap;

public enum DatasetType {
    NLMCHEM("nlmchem") ,
    CHEMDNER("chemdner"),
    CUSTOM("custom");

    private final String value;
    private static final HashMap<String, DatasetType> NameToType = new HashMap<>();

    static {
        for (DatasetType e : values()) {
            NameToType.put(e.value, e);
        }
    }

    DatasetType(String name) {
        this.value = name;
    }

    public static DatasetType ValueOf(String str){
        return NameToType.get(str);
    }
}
