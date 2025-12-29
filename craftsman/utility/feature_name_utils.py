from category_encoders import \
    BaseNEncoder, BinaryEncoder, CatBoostEncoder,\
    CountEncoder, HashingEncoder, LeaveOneOutEncoder
from sklearn.preprocessing import \
    KBinsDiscretizer, LabelEncoder, MinMaxScaler
from craftsman.utility.training_helper import \
    CraftsmanBaseNEncoder, CraftsmanBinaryEncoder, CraftsmanCatBoostEncoder,\
    CraftsmanCountEncoder, CraftsmanHashingEncoder, CraftsmanKBinsDiscretizer,\
    CraftsmanLabelEncoder, CraftsmanLeaveOneOutEncoder

def _get_set_feature_names_in(encoder, feature_names_in_: list[str] | None) -> list[str]:
    if feature_names_in_ is not None:
        if hasattr(encoder, 'n_features_in_'):
            assert len(feature_names_in_) == encoder.n_features_in_
        encoder.feature_names_in_ = feature_names_in_
    else:
        assert hasattr(encoder, 'feature_names_in_')
        feature_names_in_ = encoder.feature_names_in_
    return feature_names_in_

def _get_set_feature_names_out(encoder, feature_names_in_: list[str]) -> list[str]:
    feature_names_out_ = list(encoder.get_feature_names_out(feature_names_in_))
    encoder.feature_names_out_ = feature_names_out_
    return feature_names_out_

def _get_set_feature_names_in_out_for_BaseNEncoder(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [BaseNEncoder, CraftsmanBaseNEncoder]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

# BinarizerDiscretizer TODO?

def _get_set_feature_names_in_out_for_BinaryEncoder(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [BinaryEncoder, CraftsmanBinaryEncoder]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

def _get_set_feature_names_in_out_for_CatBoostEncoder(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [CatBoostEncoder, CraftsmanCatBoostEncoder]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

def _get_set_feature_names_in_out_for_CountEncoder(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [CountEncoder, CraftsmanCountEncoder]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

def _get_set_feature_names_in_out_for_HashingEncoder(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [HashingEncoder, CraftsmanHashingEncoder]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

def _get_set_feature_names_in_out_for_KBinsDiscretizer(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [KBinsDiscretizer, CraftsmanKBinsDiscretizer]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

def _get_set_feature_names_in_out_for_KBinsDiscretizer(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [KBinsDiscretizer, CraftsmanKBinsDiscretizer]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

def _get_set_feature_names_in_out_for_LabelEncoder(encoder, feature_names_in_: list[str]) -> list[str]:
    assert type(encoder) in [LabelEncoder, CraftsmanLabelEncoder]
    assert len(feature_names_in_) == 1
    encoder.feature_names_in_ = feature_names_in_
    encoder.feature_names_out_ = feature_names_in_
    return feature_names_in_

def _get_set_feature_names_in_out_for_LeaveOneOutEncoder(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [LeaveOneOutEncoder, CraftsmanLeaveOneOutEncoder]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

def _get_set_feature_names_in_out_for_MinMaxScaler(encoder, feature_names_in_: list[str] | None) -> list[str]:
    assert type(encoder) in [MinMaxScaler]
    feature_names_in_ = _get_set_feature_names_in(encoder, feature_names_in_)
    return _get_set_feature_names_out(encoder, feature_names_in_)

mapping = {
    BaseNEncoder: _get_set_feature_names_in_out_for_BaseNEncoder,
    CraftsmanBaseNEncoder: _get_set_feature_names_in_out_for_BaseNEncoder,
    BinaryEncoder: _get_set_feature_names_in_out_for_BinaryEncoder,
    CraftsmanBinaryEncoder: _get_set_feature_names_in_out_for_BinaryEncoder,
    CatBoostEncoder: _get_set_feature_names_in_out_for_CatBoostEncoder,
    CraftsmanCatBoostEncoder: _get_set_feature_names_in_out_for_CatBoostEncoder,
    CountEncoder: _get_set_feature_names_in_out_for_CountEncoder,
    CraftsmanCountEncoder: _get_set_feature_names_in_out_for_CountEncoder,
    HashingEncoder: _get_set_feature_names_in_out_for_HashingEncoder,
    CraftsmanHashingEncoder: _get_set_feature_names_in_out_for_HashingEncoder,
    KBinsDiscretizer: _get_set_feature_names_in_out_for_KBinsDiscretizer,
    CraftsmanKBinsDiscretizer: _get_set_feature_names_in_out_for_KBinsDiscretizer,
    LabelEncoder: _get_set_feature_names_in_out_for_LabelEncoder,
    CraftsmanLabelEncoder: _get_set_feature_names_in_out_for_LabelEncoder,
    LeaveOneOutEncoder: _get_set_feature_names_in_out_for_LeaveOneOutEncoder,
    CraftsmanLeaveOneOutEncoder: _get_set_feature_names_in_out_for_LeaveOneOutEncoder,
    MinMaxScaler: _get_set_feature_names_in_out_for_MinMaxScaler,
}

def get_set_feature_names_in_out(encoder, feature_names_in_: list[str] | None = None) -> list[str]:
    return mapping[type(encoder)](encoder, feature_names_in_)
